from pathlib import Path

from t2dsim_ai.options import default_seq_len
from t2dsim_ai.train import train_digital_twin

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    train_digital_twin(
        data_path=root / "example_model/data_example.csv",
        output_dir=root / "example_model/output",
        n_epochs=10,
        lr=1e-5,
        batch_size=16,
        seq_len=default_seq_len,
        hypo_penalization=9.0,
        hyper_penalization=90.0,
        alpha=1e-4,
        dcgm_weight=100.0,
    )
    print("Training complete. Saved to example/example_model/output/")
