import polars as pl

def main():
    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })
    print(df)

if __name__ == "__main__":
    main()
