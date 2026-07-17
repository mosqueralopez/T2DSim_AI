from pathlib import Path

from t2dsim_ai.train import train_digital_twin

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    train_digital_twin(
        data_path=root / "example_model/data_example.csv",
        output_dir=root / "example_model/output",
        n_epochs=2,
        lr=1e-5,
        batch_size=16,
    )
    print("Training complete. Artifacts saved to example/example_model/output/")
