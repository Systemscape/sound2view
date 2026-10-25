//! Live audio capture via cpal (ALSA on Linux, CoreAudio on macOS).

use std::{
    collections::VecDeque,
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
};

use cpal::{
    BufferSize, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

use crate::FrameSource;

/// Chunks buffered between the audio callback and `read` before chunks get dropped
const CHANNEL_CAPACITY: usize = 64;
/// Full scale of the I32 samples (24 bit left-justified in 32 bit, i.e. 2^31)
const I32_FULL_SCALE: f64 = 2_147_483_648.0;

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

/// Live capture from an `M`-channel input device with I32 samples
pub struct CpalSource<const M: usize> {
    /// Keeps the capture running; capture stops when dropped
    _stream: cpal::Stream,
    /// Interleaved chunks from the audio callback, scaled to [-1, 1)
    rx: Receiver<Vec<f64>>,
    /// Samples received but not yet returned by [`read`](Self::read), interleaved
    pending: VecDeque<f64>,
    /// Chunks dropped because `read` did not keep up
    dropped: Arc<AtomicUsize>,
    sample_rate: f64,
}

impl<const M: usize> CpalSource<M> {
    /// Open the first input device whose id contains `id_pattern` and start capturing.
    ///
    /// - `id_pattern`: e.g. `"hw:CARD=UMA16v2"`, see [`probe`]
    /// - `sample_rate`: in Hz, must be supported by the device
    pub fn new(id_pattern: &str, sample_rate: u32) -> Result<Self, Box<dyn Error>> {
        let host = cpal::default_host();
        let device = host
            .input_devices()?
            .find(|d| d.id().is_ok_and(|id| id.to_string().contains(id_pattern)))
            .ok_or_else(|| format!("no input device matching {id_pattern:?}"))?;

        let config = StreamConfig {
            channels: M as u16,
            sample_rate,
            buffer_size: BufferSize::Default,
        };

        let (tx, rx): (SyncSender<Vec<f64>>, _) = sync_channel(CHANNEL_CAPACITY);
        let dropped = Arc::new(AtomicUsize::new(0));
        let dropped_cb = Arc::clone(&dropped);

        let stream = device.build_input_stream(
            config,
            move |data: &[i32], _: &_| {
                let chunk = data.iter().map(|&s| s as f64 / I32_FULL_SCALE).collect();
                // Never block the audio thread: drop the chunk if the reader is behind
                if let Err(TrySendError::Full(_)) = tx.try_send(chunk) {
                    dropped_cb.fetch_add(1, Ordering::Relaxed);
                }
            },
            |e| eprintln!("capture error: {e}"),
            None,
        )?;
        stream.play()?;

        Ok(Self {
            _stream: stream,
            rx,
            pending: VecDeque::new(),
            dropped,
            sample_rate: sample_rate as f64,
        })
    }

    /// Number of chunks dropped so far because `read` did not keep up
    pub fn dropped(&self) -> usize {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl<const M: usize> FrameSource<M> for CpalSource<M> {
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Blocks until at least one frame is available, then fills as much of `buf` as possible.
    ///
    /// Never returns `Ok(0)` while the stream runs; returns an error if the stream stopped.
    fn read(&mut self, buf: &mut [f64]) -> std::io::Result<usize> {
        debug_assert!(buf.len().is_multiple_of(M));

        // If `pending` is empty: block on `self.rx.recv()` and append the chunk.
        // `recv` fails only if the callback was dropped: map to an io::Error.
        if self.pending.is_empty() {
            let chunk = self.rx.recv().map_err(std::io::Error::other)?;
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
