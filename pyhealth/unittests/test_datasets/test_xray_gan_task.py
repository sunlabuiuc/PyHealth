from pyhealth.tasks.xray_gan_task import evaluate_predictions

def test_evaluation_accuracy():
    y_true = ["positive", "negative", "positive"]
    y_pred = ["positive", "positive", "negative"]
    metrics = evaluate_predictions(y_true, y_pred)
    assert "accuracy" in metrics["weighted avg"]
