# IP Camera Setup Guide

MonitaQC supports both USB cameras and IP cameras (RTSP/HTTP streams). You can use them separately or mix them together.

## Supported Formats

### RTSP Streams
```
rtsp://username:password@192.168.1.100:554/stream1
rtsp://192.168.1.100:554/Streaming/Channels/101
```

### HTTP/MJPEG Streams
```
http://192.168.1.100/video.mjpg
http://192.168.1.100:8080/video
https://192.168.1.100/mjpg/video.mjpg
```

## Configuration

### Option 1: Environment Variable (Recommended)

Add IP cameras to your `docker-compose.yml`:

```yaml
environment:
  # Multiple IP cameras (comma-separated)
  - IP_CAMERAS=rtsp://admin:password@192.168.1.100:554/stream1,rtsp://admin:password@192.168.1.101:554/stream1
```

### Option 2: Mixed USB + IP Cameras

```yaml
environment:
  # USB cameras will be auto-detected
  - CAM_1_PATH=/dev/video0
  - CAM_2_PATH=/dev/video2
  # Add IP cameras
  - IP_CAMERAS=rtsp://192.168.1.100:554/stream1,http://192.168.1.101/video.mjpg
```

### Option 3: IP Cameras Only

```yaml
environment:
  # Disable USB camera auto-detection by setting invalid paths
  - CAM_1_PATH=/dev/null
  - CAM_2_PATH=/dev/null
  - CAM_3_PATH=/dev/null
  - CAM_4_PATH=/dev/null
  # Use only IP cameras
  - IP_CAMERAS=rtsp://camera1.local/stream,rtsp://camera2.local/stream
```

## Common IP Camera URLs

### Hikvision
```
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/102  # Sub-stream
```

### Dahua
```
rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0
rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1  # Sub-stream
```

### Axis
```
rtsp://root:password@192.168.1.100/axis-media/media.amp
http://192.168.1.100/mjpg/video.mjpg
```

### Reolink
```
rtsp://admin:password@192.168.1.100:554/h264Preview_01_main
rtsp://admin:password@192.168.1.100:554/h264Preview_01_sub  # Sub-stream
```

### Foscam
```
rtsp://username:password@192.168.1.100:554/videoMain
http://192.168.1.100:88/cgi-bin/CGIStream.cgi?cmd=GetMJStream
```

### Generic ONVIF Cameras
```
rtsp://username:password@192.168.1.100:554/onvif1
rtsp://username:password@192.168.1.100:554/profile1
```

## Testing IP Camera Connection

Before adding to MonitaQC, test your camera URL:

### Using VLC Media Player
1. Open VLC
2. Media → Open Network Stream
3. Enter your RTSP/HTTP URL
4. Click Play

### Using FFmpeg
```bash
ffmpeg -i "rtsp://admin:password@192.168.1.100:554/stream1" -frames:v 1 test.jpg
```

### Using OpenCV (Python)
```python
import cv2
cap = cv2.VideoCapture("rtsp://admin:password@192.168.1.100:554/stream1")
ret, frame = cap.read()
if ret:
    cv2.imwrite("test.jpg", frame)
    print("Success!")
else:
    print("Failed to connect")
cap.release()
```

## Troubleshooting

### Connection Issues

**Problem**: Camera not connecting
- **Solution**: Check network connectivity: `ping 192.168.1.100`
- **Solution**: Verify credentials are correct
- **Solution**: Check firewall/router settings
- **Solution**: Try different stream URLs (main/sub stream)

**Problem**: Slow/laggy stream
- **Solution**: Use sub-stream instead of main stream (lower resolution)
- **Solution**: Check network bandwidth
- **Solution**: Reduce camera resolution in camera settings

**Problem**: Stream disconnects frequently
- **Solution**: Check camera keepalive settings
- **Solution**: Increase network timeout in OpenCV
- **Solution**: Use TCP instead of UDP for RTSP:
  ```yaml
  - IP_CAMERAS=rtsp://admin:password@192.168.1.100:554/stream1?tcp
  ```

### Authentication Issues

**Problem**: 401 Unauthorized
- **Solution**: URL encode special characters in password:
  - `@` → `%40`
  - `#` → `%23`
  - `&` → `%26`
  - Example: `password@123` → `password%40123`

**Problem**: Connection refused
- **Solution**: Check RTSP port (default 554, some use 8554)
- **Solution**: Enable RTSP in camera web interface
- **Solution**: Check if camera requires ONVIF authentication

## Performance Tips

1. **Use Sub-streams**: Main streams are often 4K/1080p - use 720p/480p sub-streams for processing
2. **Limit FPS**: Set camera to 15-30 FPS instead of 60 FPS
3. **Use H.264**: Avoid MJPEG streams if possible (much higher bandwidth)
4. **Local Network**: Keep cameras on same network as MonitaQC server
5. **Wired Connection**: Use Ethernet instead of Wi-Fi for cameras when possible

## Example Configuration

Full setup with 2 USB cameras and 2 IP cameras:

```yaml
services:
  monitait_vision_engine:
    environment:
      # USB cameras (auto-detected)
      - CAM_1_PATH=/dev/video0
      - CAM_2_PATH=/dev/video2

      # IP cameras (RTSP + HTTP)
      - IP_CAMERAS=rtsp://admin:pass123@192.168.1.100:554/stream1,http://192.168.1.101/video.mjpg

      # Other settings...
      - REDIS_HOST=redis
      - REDIS_PORT=6379
```

Result: 4 total cameras (video0, video2, IP camera 1, IP camera 2)

## Security Recommendations

1. **Change Default Passwords**: Never use default camera passwords
2. **Use Strong Passwords**: Mix letters, numbers, symbols
3. **Network Segmentation**: Put cameras on isolated VLAN
4. **Disable UPnP**: Prevent cameras from opening ports
5. **Update Firmware**: Keep camera firmware up to date
6. **Use HTTPS/RTSPS**: When supported by camera
7. **VPN Access**: For remote monitoring, use VPN instead of port forwarding

## Support

For issues or questions:
- Email: admin@smartfalcon-ai.com
- GitHub: https://github.com/smartfalcon-ai/MonitaQC
