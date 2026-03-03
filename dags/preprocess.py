import pandas as pd
import os

def main():

    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [0, 1, 0, 1, 0]
    })
    output_path = os.path.join(os.path.dirname(__file__), "processed.csv")
    df.to_csv(output_path, index=False)

    print("Preprocess finished")
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()