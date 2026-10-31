//! macOS backend: AVFoundation via objc2.
//!
//! An `AVCaptureSession` with an `AVCaptureVideoDataOutput` delivers frames as 32BGRA
//! (AVFoundation decodes MJPEG and YUV itself) to a delegate on a dispatch queue, which converts
//! them to RGBA and sends them to `next_frame`. Device format and frame rate are chosen from the
//! same format entry, so AVFoundation never gets an unsupported combination (it throws an
//! Objective-C exception then, which aborts the process).

use std::{
    error::Error,
    slice,
    sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel},
    time::Duration,
};

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchRetained};
use objc2::{
    AnyThread, DefinedClass, define_class, msg_send,
    rc::{Retained, autoreleasepool},
    runtime::{AnyObject, Bool, ProtocolObject},
};
use objc2_av_foundation::{
    AVAuthorizationStatus, AVCaptureConnection, AVCaptureDevice,
    AVCaptureDeviceDiscoverySession, AVCaptureDeviceFormat, AVCaptureDeviceInput,
    AVCaptureDevicePosition, AVCaptureDeviceTypeBuiltInWideAngleCamera,
    AVCaptureDeviceTypeExternal, AVCaptureOutput, AVCaptureSession, AVCaptureVideoDataOutput,
    AVCaptureVideoDataOutputSampleBufferDelegate, AVFrameRateRange, AVMediaTypeVideo,
};
use objc2_core_foundation::CFString;
use objc2_core_media::{CMSampleBuffer, CMTime, CMVideoFormatDescriptionGetDimensions};
use objc2_core_video::{
    CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
    CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelBufferPixelFormatTypeKey,
    kCVPixelFormatType_32BGRA, kCVReturnSuccess,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSObject, NSObjectProtocol, NSString};

use super::{CameraEntry, VideoError};
use crate::{ImageSource, VideoFrame};

/// `next_frame` gives up after this long without a frame (e.g. camera unplugged)
const FRAME_TIMEOUT: Duration = Duration::from_secs(2);

/// Request camera access; required once before any other camera call (macOS asks the user on
/// the first run). Returns immediately, see [`is_authorized`].
pub fn initialize() {
    let handler = RcBlock::new(|granted: Bool| {
        if !granted.as_bool() {
            eprintln!("{}", VideoError::NotAuthorized);
        }
    });
    unsafe {
        AVCaptureDevice::requestAccessForMediaType_completionHandler(video_media_type(), &handler)
    };
}

/// True once camera access is granted
pub fn is_authorized() -> bool {
    let status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(video_media_type()) };
    status == AVAuthorizationStatus::Authorized
}

/// Print all cameras with their formats, one line per format and frame rate range.
///
/// The terminal needs camera access, see the README.
pub fn probe() -> Result<(), Box<dyn Error>> {
    if !is_authorized() {
        println!("{}", VideoError::NotAuthorized);
        return Ok(());
    }

    autoreleasepool(|_| {
        let devices = devices();
        if devices.is_empty() {
            println!("no cameras found");
        }
        for (index, device) in devices.iter().enumerate() {
            let name = unsafe { device.localizedName() };
            let id = unsafe { device.uniqueID() };
            println!("{index} {name} ({id})");
            for option in format_options(device) {
                let (min_fps, max_fps) = option.fps_range();
                println!(
                    "    {}x{} @ {min_fps:.0}-{max_fps:.0} FPS, {}",
                    option.width,
                    option.height,
                    fourcc(&option.format)
                );
            }
        }
    });
    Ok(())
}

/// All cameras, in discovery order (the index of [`CameraEntry`])
pub fn list_cameras() -> Result<Vec<CameraEntry>, VideoError> {
    if !is_authorized() {
        return Err(VideoError::NotAuthorized);
    }
    let cameras: Vec<CameraEntry> = autoreleasepool(|_| {
        devices()
            .iter()
            .enumerate()
            .map(|(index, device)| CameraEntry {
                index: index as u32,
                name: unsafe { device.localizedName() }.to_string(),
            })
            .collect()
    });
    Ok(cameras)
}

/// Live camera capture, frames converted to RGBA
pub struct CameraSource {
    session: Retained<AVCaptureSession>,
    rx: Receiver<VideoFrame>,
    // Kept alive while the session runs
    _output: Retained<AVCaptureVideoDataOutput>,
    _delegate: Retained<FrameDelegate>,
    _queue: DispatchRetained<DispatchQueue>,
}

impl CameraSource {
    /// Open camera `index` (see [`list_cameras`]) and start streaming, with the device format
    /// closest to `width` x `height` (exact if listed) whose frame rate range is closest to
    /// `fps`. A different resolution is printed, because the field of view in the config
    /// belongs to the requested one.
    pub fn new(index: u32, width: u32, height: u32, fps: u32) -> Result<Self, VideoError> {
        if !is_authorized() {
            return Err(VideoError::NotAuthorized);
        }

        autoreleasepool(|_| {
            let device = devices()
                .into_iter()
                .nth(index as usize)
                .ok_or(VideoError::NoSuchIndex(index))?;

            let fps = f64::from(fps);
            let option = format_options(&device)
                .into_iter()
                .min_by(|a, b| {
                    let (a, b) = (a.distance(width, height, fps), b.distance(width, height, fps));
                    a.0.cmp(&b.0).then(a.1.total_cmp(&b.1))
                })
                .ok_or_else(|| VideoError::Camera("camera lists no video formats".to_string()))?;
            if (option.width, option.height) != (width as i32, height as i32) {
                eprintln!(
                    "camera {index}: requested {width}x{height}, got {}x{}",
                    option.width, option.height
                );
            }

            let input =
                unsafe { AVCaptureDeviceInput::deviceInputWithDevice_error(&device) }.map_err(ns_error)?;

            // 32BGRA: AVFoundation converts from the device format, no decoding on our side
            let output = unsafe { AVCaptureVideoDataOutput::new() };
            // SAFETY: CFString and NSString are toll-free bridged
            let key: &NSString =
                unsafe { &*(kCVPixelBufferPixelFormatTypeKey as *const CFString).cast::<NSString>() };
            let pixel_format = NSNumber::numberWithUnsignedInt(kCVPixelFormatType_32BGRA);
            let value: &AnyObject = &pixel_format;
            let settings = NSDictionary::from_slices(&[key], &[value]);
            unsafe {
                output.setVideoSettings(Some(&settings));
                output.setAlwaysDiscardsLateVideoFrames(true);
            }

            let (tx, rx) = sync_channel(1);
            let delegate = FrameDelegate::new(tx);
            let queue = DispatchQueue::new("sound2view.camera", None);
            unsafe {
                output.setSampleBufferDelegate_queue(
                    Some(ProtocolObject::from_ref(&*delegate)),
                    Some(&queue),
                );
            }

            let session = unsafe { AVCaptureSession::new() };
            unsafe {
                session.beginConfiguration();
                if !session.canAddInput(&input) || !session.canAddOutput(&output) {
                    session.commitConfiguration();
                    return Err(VideoError::Camera(
                        "camera cannot be added to a capture session".to_string(),
                    ));
                }
                session.addInput(&input);
                session.addOutput(&output);

                // Setting the active format switches the session to the input-priority preset,
                // so the session keeps it instead of choosing its own
                if let Err(e) = device.lockForConfiguration() {
                    session.commitConfiguration();
                    return Err(ns_error(e));
                }
                device.setActiveFormat(&option.format);
                let duration = option.frame_duration(fps);
                device.setActiveVideoMinFrameDuration(duration);
                device.setActiveVideoMaxFrameDuration(duration);
                session.commitConfiguration();
                session.startRunning();
                device.unlockForConfiguration();
            }

            Ok(Self {
                session,
                rx,
                _output: output,
                _delegate: delegate,
                _queue: queue,
            })
        })
    }
}

impl Drop for CameraSource {
    fn drop(&mut self) {
        unsafe { self.session.stopRunning() };
    }
}

impl ImageSource for CameraSource {
    type Error = VideoError;

    /// Blocks until the next frame; error after `FRAME_TIMEOUT` without one.
    fn next_frame(&mut self) -> Result<VideoFrame, VideoError> {
        self.rx.recv_timeout(FRAME_TIMEOUT).map_err(|e| match e {
            RecvTimeoutError::Timeout => VideoError::NoFrame(FRAME_TIMEOUT),
            RecvTimeoutError::Disconnected => VideoError::Closed,
        })
    }
}

define_class!(
    // SAFETY: NSObject has no subclassing requirements, and `FrameDelegate` does not implement
    // `Drop`.
    #[unsafe(super(NSObject))]
    #[name = "Sound2ViewFrameDelegate"]
    #[ivars = SyncSender<VideoFrame>]
    struct FrameDelegate;

    unsafe impl NSObjectProtocol for FrameDelegate {}

    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for FrameDelegate {
        /// Called on the dispatch queue for every frame
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        fn capture_output(
            &self,
            _output: &AVCaptureOutput,
            sample_buffer: &CMSampleBuffer,
            _connection: &AVCaptureConnection,
        ) {
            // SAFETY: the sample buffer is valid during the callback
            if let Some(frame) = unsafe { to_rgba(sample_buffer) } {
                // Drop the frame while the reader has not taken the previous one; a closed
                // channel is noticed by `next_frame`
                let _ = self.ivars().try_send(frame);
            }
        }
    }
);

impl FrameDelegate {
    fn new(tx: SyncSender<VideoFrame>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(tx);
        unsafe { msg_send![super(this), init] }
    }
}

/// RGBA copy of a 32BGRA sample buffer; `None` for another pixel format or a failed lock.
///
/// # Safety
/// `sample_buffer` must be valid for the duration of the call.
unsafe fn to_rgba(sample_buffer: &CMSampleBuffer) -> Option<VideoFrame> {
    let buffer = unsafe { sample_buffer.image_buffer() }?;
    if CVPixelBufferGetPixelFormatType(&buffer) != kCVPixelFormatType_32BGRA {
        return None;
    }
    let lock = CVPixelBufferLockFlags::ReadOnly;
    if unsafe { CVPixelBufferLockBaseAddress(&buffer, lock) } != kCVReturnSuccess {
        return None;
    }

    let width = CVPixelBufferGetWidth(&buffer);
    let height = CVPixelBufferGetHeight(&buffer);
    let stride = CVPixelBufferGetBytesPerRow(&buffer);
    let base = CVPixelBufferGetBaseAddress(&buffer).cast::<u8>();

    let mut pixels = Vec::with_capacity(width * height * 4);
    if !base.is_null() {
        for row in 0..height {
            // SAFETY: the locked buffer holds `height` rows of `stride` bytes, each starting
            // with `width` BGRA pixels
            let line = unsafe { slice::from_raw_parts(base.add(row * stride), width * 4) };
            for bgra in line.chunks_exact(4) {
                pixels.extend_from_slice(&[bgra[2], bgra[1], bgra[0], 255]);
            }
        }
    }
    unsafe { CVPixelBufferUnlockBaseAddress(&buffer, lock) };

    (!base.is_null()).then_some(VideoFrame {
        width: width as u32,
        height: height as u32,
        pixels,
    })
}

/// One frame rate range of one device format
struct FormatOption {
    format: Retained<AVCaptureDeviceFormat>,
    width: i32,
    height: i32,
    range: Retained<AVFrameRateRange>,
}

impl FormatOption {
    fn fps_range(&self) -> (f64, f64) {
        unsafe { (self.range.minFrameRate(), self.range.maxFrameRate()) }
    }

    /// Sort key against the request: resolution mismatch in pixels, then how far `fps` lies
    /// outside the frame rate range
    fn distance(&self, width: u32, height: u32, fps: f64) -> (u64, f64) {
        let pixels = self.width.abs_diff(width as i32) + self.height.abs_diff(height as i32);
        let (min_fps, max_fps) = self.fps_range();
        let fps_off = (min_fps - fps).max(fps - max_fps).max(0.0);
        (u64::from(pixels), fps_off)
    }

    /// Frame duration for `fps`, clamped to the range; the range's own durations at its ends
    /// (exact even for rates like 29.97)
    fn frame_duration(&self, fps: f64) -> CMTime {
        let (min_fps, max_fps) = self.fps_range();
        unsafe {
            if fps >= max_fps {
                self.range.minFrameDuration()
            } else if fps <= min_fps {
                self.range.maxFrameDuration()
            } else {
                CMTime::new(1, fps.round() as i32)
            }
        }
    }
}

/// Cameras in discovery order: external (USB) and built-in
fn devices() -> Vec<Retained<AVCaptureDevice>> {
    unsafe {
        let types = NSArray::from_slice(&[
            AVCaptureDeviceTypeExternal,
            AVCaptureDeviceTypeBuiltInWideAngleCamera,
        ]);
        let discovery = AVCaptureDeviceDiscoverySession::discoverySessionWithDeviceTypes_mediaType_position(
            &types,
            Some(video_media_type()),
            AVCaptureDevicePosition::Unspecified,
        );
        discovery.devices().to_vec()
    }
}

/// All frame rate ranges of all formats of `device`
fn format_options(device: &AVCaptureDevice) -> Vec<FormatOption> {
    let mut options = Vec::new();
    for format in unsafe { device.formats() }.to_vec() {
        let description = unsafe { format.formatDescription() };
        let dimensions = unsafe { CMVideoFormatDescriptionGetDimensions(&description) };
        for range in unsafe { format.videoSupportedFrameRateRanges() }.to_vec() {
            options.push(FormatOption {
                format: format.clone(),
                width: dimensions.width,
                height: dimensions.height,
                range,
            });
        }
    }
    options
}

/// Pixel format of a device format as four characters, e.g. "420v" or "yuvs"
fn fourcc(format: &AVCaptureDeviceFormat) -> String {
    let code = unsafe { format.formatDescription().media_sub_type() };
    code.to_be_bytes().iter().map(|&b| b as char).collect()
}

fn video_media_type() -> &'static NSString {
    unsafe { AVMediaTypeVideo }.expect("AVMediaTypeVideo exists on macOS")
}

fn ns_error(error: Retained<NSError>) -> VideoError {
    VideoError::Camera(error.localizedDescription().to_string())
}
