# sound2view

Acoustic camera with the miniDSP UMA-16 v2.

## Linux setup

Ubuntu/Debian: install `alsa-sys` with
```sh
sudo apt install libasound2-dev
```

Camera access (V4L2) needs the user in the `video` group:
```sh
sudo usermod -aG video $USER
```
Log out and back in (or reboot) for the group to take effect; check with `groups`.

## Usage

```sh
cargo run --release -- probe      # list audio input devices
cargo run --release -- camprobe   # list cameras and formats
cargo run --release -- camshot    # time camera frames, save camshot.png
cargo run --release -- gui        # live window with the camera image
cargo run --release -- mic        # live maps from the UMA-16
cargo run --release -- scene.toml # simulated scene (default: config.toml)
```

Settings (audio device, camera name and format, field of view, band, beamformer, overlay range)
are read from `sound2view.toml` in the working directory; every entry is optional and defaults
to the values in the shipped file. Beamformer, nu, band and overlay range can also be changed
live in the GUI (not written back to the file).
