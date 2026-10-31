//! Live audio capture via cpal (ALSA on Linux, CoreAudio on macOS).

use std::{
    collections::VecDeque,
    error::Error,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel},
    },
    time::Duration,
};

use cpal::{
    BufferSize, ErrorKind, FromSample, SampleFormat, SizedSample, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

use crate::FrameSource;

/// Chunks buffered between the audio callback and `read` before chunks get dropped
const CHANNEL_CAPACITY: usize = 64;

/// Case-insensitive patterns that identify the UMA-16 in device ids or descriptions
const UMA16_PATTERNS: [&str; 2] = ["uma", "minidsp"];
/// Devices with more configs than this are converting (plug) devices; their configs are not printed
const MAX_CONFIGS: usize = 20;

/// Print the UMA-16 input devices with their supported configs, one line per config.
///
/// If no device matches [`UMA16_PATTERNS`], print only the ids of all input devices.
/// ALSA prints errors for unusable devices to stderr; hide them with `2>/dev/null`.
pub fn probe() -> Result<(), Box<dyn Error>> {
    for host_id in cpal::available_hosts() {
        println!("Host: {}", host_id.name());
        let host = cpal::host_from_id(host_id)?;

        let mut others = Vec::new();
        let mut found = false;

        for device in host.input_devices()? {
            let id = device
                .id()
                .map_or("unknown id".to_string(), |id| id.to_string());
            let desc = device
                .description()
                .map_or(String::new(), |desc| desc.to_string());

            let text = format!("{id} {desc}").to_lowercase();
            if !UMA16_PATTERNS.iter().any(|p| text.contains(p)) {
                others.push(id);
                continue;
            }

            found = true;
            println!("  {id} ({desc})");
            match device.supported_input_configs() {
                Ok(configs) => {
                    let configs: Vec<_> = configs.collect();
                    // Plug devices convert to every channel count and format; not useful here
                    if configs.len() > MAX_CONFIGS {
                        println!("    {} configs (converting device), skipped", configs.len());
                        continue;
                    }
                    for c in configs {
                        println!(
                            "    {} ch, {}..{} Hz, {:?}",
                            c.channels(),
                            c.min_sample_rate(),
                            c.max_sample_rate(),
                            c.sample_format()
                        );
                    }
                }
                Err(e) => println!("    error getting input configs: {e}"),
            }
        }

        if !found {
            println!("  no UMA-16 found, all input devices:");
            for id in others {
                println!("    {id}");
            }
        }
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("no audio input device matching {0:?}")]
    NotFound(String),
    #[error("audio device: {0}")]
    Device(#[from] cpal::Error),
    #[error("audio stream: {0}")]
    Stream(cpal::Error),
    #[error("no audio data for {} s", .0.as_secs_f64())]
    NoData(Duration),
    #[error("audio stream closed")]
    Closed,
    #[error("unsupported sample format {0}")]
    UnsupportedFormat(SampleFormat),
}

/// `read` gives up after this long without data (e.g. device unplugged without an error)
const NO_DATA_TIMEOUT: Duration = Duration::from_secs(2);
/// `read` checks for stream errors at least this often while waiting
const POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Live capture from an `M`-channel input device
pub struct CpalSource<const M: usize> {
    /// Keeps the capture running; capture stops when dropped
    _stream: cpal::Stream,
    /// Interleaved chunks from the audio callback, scaled to [-1, 1)
    rx: Receiver<Vec<f64>>,
    /// Samples received but not yet returned by [`read`](Self::read), interleaved
    pending: VecDeque<f64>,
    /// Chunks dropped because `read` did not keep up
    dropped: Arc<AtomicUsize>,
    /// First fatal error reported by the stream's error callback
    stream_error: Arc<Mutex<Option<cpal::Error>>>,
    sample_rate: f64,
}

impl<const M: usize> CpalSource<M> {
    /// Open the first input device whose id or name contains `pattern` and start capturing.
    ///
    /// - `pattern`: e.g. `"hw:CARD=UMA16v2"` (ALSA lists the UMA-16 several times, only `hw:` is
    ///   the raw device) or `"UMA16"` (CoreAudio), see [`probe`]
    /// - `sample_rate`: in Hz, must be supported by the device
    ///
    /// The stream uses the device's default sample format (I32 for the UMA-16 on ALSA, F32 on
    /// CoreAudio), converted to f64 in [-1, 1).
    pub fn new(pattern: &str, sample_rate: u32) -> Result<Self, CaptureError> {
        let host = cpal::default_host();
        let device = host
            .input_devices()?
            .find(|d| {
                let id = d.id().map(|id| id.to_string()).unwrap_or_default();
                let name = d.description().map(|n| n.to_string()).unwrap_or_default();
                id.contains(pattern) || name.contains(pattern)
            })
            .ok_or_else(|| CaptureError::NotFound(pattern.to_string()))?;

        let config = StreamConfig {
            channels: M as u16,
            sample_rate,
            buffer_size: BufferSize::Default,
        };

        let (tx, rx) = sync_channel(CHANNEL_CAPACITY);
        let dropped = Arc::new(AtomicUsize::new(0));
        let stream_error = Arc::new(Mutex::new(None));
        let callbacks = Callbacks {
            tx,
            dropped: Arc::clone(&dropped),
            stream_error: Arc::clone(&stream_error),
        };

        let stream = match device.default_input_config()?.sample_format() {
            SampleFormat::I32 => build_stream::<i32>(&device, config, callbacks)?,
            SampleFormat::F32 => build_stream::<f32>(&device, config, callbacks)?,
            SampleFormat::I16 => build_stream::<i16>(&device, config, callbacks)?,
            format => return Err(CaptureError::UnsupportedFormat(format)),
        };
        stream.play()?;

        Ok(Self {
            _stream: stream,
            rx,
            pending: VecDeque::new(),
            dropped,
            stream_error,
            sample_rate: sample_rate as f64,
        })
    }

    /// Number of chunks dropped so far because `read` did not keep up
    pub fn dropped(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Block until the next chunk arrives; error on a stream error, a closed stream or
    /// `NO_DATA_TIMEOUT` without data.
    fn next_chunk(&self) -> Result<Vec<f64>, CaptureError> {
        let mut waited = Duration::ZERO;
        loop {
            if let Some(e) = self.stream_error.lock().unwrap().clone() {
                return Err(CaptureError::Stream(e));
            }
            match self.rx.recv_timeout(POLL_INTERVAL) {
                Ok(chunk) => return Ok(chunk),
                Err(RecvTimeoutError::Disconnected) => return Err(CaptureError::Closed),
                Err(RecvTimeoutError::Timeout) => {
                    waited += POLL_INTERVAL;
                    if waited >= NO_DATA_TIMEOUT {
                        return Err(CaptureError::NoData(waited));
                    }
                }
            }
        }
    }
}

/// Shared state of the stream callbacks
struct Callbacks {
    /// Interleaved chunks for `read`
    tx: SyncSender<Vec<f64>>,
    dropped: Arc<AtomicUsize>,
    stream_error: Arc<Mutex<Option<cpal::Error>>>,
}

/// Input stream with sample type `T`; each callback buffer is converted to f64 in [-1, 1)
/// (integers divided by their full scale, e.g. 2^31 for I32) and sent as one chunk.
fn build_stream<T>(
    device: &cpal::Device,
    config: StreamConfig,
    callbacks: Callbacks,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample + Send + 'static,
    f64: FromSample<T>,
{
    let Callbacks {
        tx,
        dropped,
        stream_error,
    } = callbacks;

    device.build_input_stream(
        config,
        move |data: &[T], _: &_| {
            let chunk = data.iter().map(|&s| s.to_sample::<f64>()).collect();
            // Never block the audio thread: drop the chunk if the reader is behind
            if let Err(TrySendError::Full(_)) = tx.try_send(chunk) {
                dropped.fetch_add(1, Ordering::Relaxed);
            }
        },
        move |e: cpal::Error| match e.kind() {
            // Glitches, the stream keeps running
            ErrorKind::Xrun | ErrorKind::RealtimeDenied | ErrorKind::DeviceChanged => {
                eprintln!("audio stream: {e}");
            }
            // E.g. device unplugged: keep the first error for `read`
            _ => {
                stream_error.lock().unwrap().get_or_insert(e);
            }
        },
        None,
    )
}

impl<const M: usize> FrameSource<M> for CpalSource<M> {
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Blocks until at least one frame is available, then fills as much of `buf` as possible.
    ///
    /// Never returns `Ok(0)` while the stream runs. A [`CaptureError`] (stream error, stream
    /// closed, no data for `NO_DATA_TIMEOUT`) is returned wrapped in an `io::Error`.
    fn read(&mut self, buf: &mut [f64]) -> std::io::Result<usize> {
        debug_assert!(buf.len().is_multiple_of(M));

        if self.pending.is_empty() {
            let chunk = self.next_chunk().map_err(std::io::Error::other)?;
            debug_assert!(chunk.len().is_multiple_of(M));
            self.pending.extend(chunk);
        }

        // Copy whole frames (multiples of M samples) from `pending` into `buf`.
        let (front, back) = self.pending.as_slices();

        let total = buf.len().min(self.pending.len()); // both multiples of M
        let n_front = front.len().min(total);
        let n_back = total - n_front;

        buf[..n_front].copy_from_slice(&front[..n_front]);
        buf[n_front..total].copy_from_slice(&back[..n_back]);

        self.pending.drain(..total);

        // Return the number of frames copied, not the number of elements
        Ok(total / M)
    }
}
