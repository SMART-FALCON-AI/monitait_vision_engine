# USB Webcam Setup for WSL2

This guide explains how to use your Windows webcam in WSL2 for testing USB cameras in MonitaQC.

## Prerequisites
- Windows 10/11 with WSL2
- Docker Desktop for Windows
- Administrator access

## Step 1: Install USB/IP for Windows

Open **PowerShell as Administrator** and run:

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

Or download directly from: https://github.com/dorssel/usbipd-win/releases

After installation, **restart PowerShell as Administrator**.

## Step 2: Install USB/IP Tools in WSL

Open your WSL terminal:

```bash
sudo apt update
sudo apt install linux-tools-generic hwdata
sudo update-alternatives --install /usr/local/bin/usbip usbip /usr/lib/linux-tools/*/usbip 20
```

## Step 3: Find Your Webcam

In **PowerShell as Administrator**:

```powershell
usbipd list
```

Example output:
```
BUSID  VID:PID    DEVICE                          STATE
1-4    046d:0825  Logitech Webcam C270            Not shared
2-1    8087:0032  Intel Wireless Bluetooth        Not shared
```

Note your webcam's **BUSID** (e.g., `1-4`).

## Step 4: Share the Webcam

In **PowerShell as Administrator**:

```powershell
# Replace 1-4 with your webcam's BUSID
usbipd bind --busid 1-4
```

This only needs to be done **once** per device.

## Step 5: Attach to WSL

**Every time you restart** Windows/WSL, run this in **PowerShell as Administrator**:

```powershell
# Replace 1-4 with your webcam's BUSID
usbipd attach --wsl --busid 1-4
```

## Step 6: Verify in WSL

In your WSL terminal:

```bash
ls -l /dev/video*
```

You should see:
```
crw-rw---- 1 root video 81, 0 Dec 30 08:00 /dev/video0
crw-rw---- 1 root video 81, 1 Dec 30 08:00 /dev/video1
```

> **Note:** Even numbers (video0, video2) are typically the actual camera, odd numbers are metadata streams.

## Step 7: Start MonitaQC with USB Camera

The docker-compose.yml is already configured to use `/dev/video0`:

```bash
cd /c/projects/MonitaQC
docker compose down
docker compose up -d
```

## Step 8: Verify Camera in MonitaQC

1. Open http://localhost:5050/status
2. Look for **Camera Monitoring** section
3. You should see:
   - Camera 1 (USB) - your webcam
   - Camera 2 (IP) - your IP camera 192.168.0.108

## Troubleshooting

### Webcam Not Appearing in WSL

```bash
# Check if USB device is attached
lsusb

# Check kernel messages
dmesg | grep video

# Check video devices
v4l2-ctl --list-devices
```

### Permission Issues

```bash
# Add your user to video group
sudo usermod -aG video $USER

# Log out and log back in, then check
groups
```

### Detach Webcam from WSL

In **PowerShell as Administrator**:

```powershell
usbipd detach --busid 1-4
```

### Docker Can't Access Camera

Make sure:
1. Camera is attached in WSL (`ls /dev/video*` shows it)
2. Docker Desktop WSL integration is enabled
3. Container has correct device mapping (`devices: - /dev:/dev`)
4. Container runs in privileged mode (`privileged: true`)

## Quick Reference Commands

```powershell
# Windows PowerShell (as Admin) - List devices
usbipd list

# Windows PowerShell (as Admin) - Attach webcam
usbipd attach --wsl --busid 1-4

# Windows PowerShell (as Admin) - Detach webcam
usbipd detach --busid 1-4

# WSL - Check video devices
ls -l /dev/video*

# WSL - Test camera with OpenCV
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"

# WSL - Restart MonitaQC
cd /c/projects/MonitaQC && docker compose restart monitait_vision_engine
```

## Camera Configuration

Once the USB camera is working, you can configure it via the web interface:

1. **Exposure Control** - Adjust camera exposure (1-10000)
2. **Gain Control** - Adjust camera gain (0-255)
3. **Brightness** - Adjust camera brightness (0-255)
4. **ROI (Region of Interest)** - Define capture area

These controls work for USB cameras but **not for IP cameras** (RTSP streams).

## Notes

- You need to run `usbipd attach` **every time** you restart Windows or WSL
- The webcam will be unavailable in Windows while attached to WSL
- Multiple /dev/videoX devices may appear (use even numbers: 0, 2, 4)
- USB cameras support all OpenCV controls (exposure, gain, brightness)
- IP cameras don't support these controls - configure via camera web interface

## Alternative: Test with IP Camera Only

Since you already have a working IP camera (192.168.0.108), you can test all MonitaQC functionality without USB cameras. The system works identically with both camera types for:
- Live monitoring
- Capture triggers
- YOLO detection
- Quality control checks

USB cameras would be added later when deployed on production hardware.
