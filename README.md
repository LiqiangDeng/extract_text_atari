# Extract_text_atari

## Overview

The project is mainly used for ML/RL research, by downloading atari's game videos from YouTube and dynamically extracting scores as rewards, providing preliminary preparation for future ML/RL research. Currently, only preliminary code is available.

We are conducting research by extracting frame and score information in bulk from the game dataset.

## Start
```bash
pip install

python download_video.py {Youtube_video_url}

# The code is being improved and needs to be manually modified using appropriate detection methods
python auto_extract_text.py {Video_file_name}
```

```base
python main.py
```

## Reference
[Recurrent Visual Attention code implementation](https://github.com/kevinzakka/recurrent-visual-attention)