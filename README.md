Auto BPM Sync for TouchDesigner

Automatic BPM detection and synchronization for TouchDesigner, powered by a Temporal Convolutional Network trained on 7,000+ tracks spanning jazz, techno, footwork, and more.

![Auto BPM Sync demo](img/autobpmsync_screenshot.png)

## Setup

1. Open the project and enter the `AutoBpm` container.
2. Locate the `tdPyEnvManager` palette component.
3. In its parameters, create a new Python environment:
   - Mode: Python vEnv
   - Source: `requirements.txt`

## Usage

After the environment is created, the system runs automatically inside the `AutoBpm` container.

### Controls

- Reset (Momentary): Restarts the BPM detector.
- Sync Tempo (Momentary): Sets the project file's tempo to the currently detected BPM.
- Autosync (Toggle Down): When enabled, the project file's tempo automatically syncs to the detected BPM.