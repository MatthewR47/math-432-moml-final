# GPS Tracking App

This GPS tracking app is designed to evaluate positions based on satellite data using advanced mathematical techniques such as Least Squares and Singular Value Decomposition (SVD). The application aims to provide accurate location tracking by processing satellite signals.

## Features

- **Least Squares Method**: Implements a class for performing least squares calculations to evaluate positions based on satellite data.
- **SVD Method**: Implements a class for performing Singular Value Decomposition to enhance position evaluation.
- **User-Friendly Interface**: The main application allows users to input satellite data and receive position estimates.

## Installation

To get started with the GPS tracking app, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gps-tracking-app.git
   ```
2. Navigate to the project directory:
   ```
   cd gps-tracking-app
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the GPS tracking app, execute the following command:

```
python src/main.py
```

Follow the prompts to input satellite data and receive position estimates.

## Testing

To run the unit tests for the Least Squares and SVD implementations, use the following command:

```
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.