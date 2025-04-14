# Third Umpire Decision Automation using AI

This project aims to automate third umpire decisions in cricket using artificial intelligence. The application provides features to detect no balls, runouts/stumpings, and caught behinds by analyzing video footage of cricket matches.

## Features

1. **No Ball Detection**: 
   - Utilizes a YOLO model to determine if a delivery is a no ball or a legal ball.
   - Processes video frames and declares a no ball if detected at any point.

2. **Runout/Stumping Detection**: 
   - Employs two YOLO models: one to check if the ball has hit the stumps and another to determine if the batsman is inside the crease.
   - Analyzes video frames and makes a decision based on the position of the batsman when the stumps are hit.

3. **Caught Behind Detection**: 
   - Uses a YOLO model to check for an edge between the bat and the ball.
   - Determines if the batsman is out based on the detection of an edge.

## Project Structure

```
third-umpire-decision
├── app
│   ├── __init__.py
│   ├── routes.py
│   ├── static
│   │   └── styles.css
│   ├── templates
│   │   └── index.html
│   └── models
│       ├── no_ball_model.py
│       ├── runout_stumping_model.py
│       └── caught_behind_model.py
├── uploads
├── requirements.txt
├── run.py
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/rohitsarje17/AI3rdUmp.git
   cd third-umpire-decision
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python run.py
   ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Upload a video file and select the feature you want to apply (No Ball, Runout/Stumping, Caught Behind).

4. The application will process the video and display the decision based on the selected feature.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.
