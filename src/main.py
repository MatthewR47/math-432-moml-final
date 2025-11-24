import sys
from gps.least_squares import LeastSquares
from gps.svd import SVD

def main():
    
    # Example satellite data (latitude, longitude, distance)
    # TODO use a different format or source
    satellite_data = [
        (34.0522, -118.2437, 1000),
        (36.1699, -115.1398, 1500),
        (40.7128, -74.0060, 2000)
    ]
    
    # Initialize the position evaluators
    ls_evaluator = LeastSquares()
    svd_evaluator = SVD()
    
    # Fit the model using least squares
    position_ls = ls_evaluator.fit(satellite_data)
    print(f"Estimated Position using Least Squares: {position_ls}")
    
    # Perform SVD decomposition and reconstruction
    svd_evaluator.decompose(satellite_data)
    position_svd = svd_evaluator.reconstruct()
    print(f"Estimated Position using SVD: {position_svd}")

if __name__ == "__main__":
    main()
