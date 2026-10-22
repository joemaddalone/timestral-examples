from dataclasses import dataclass

import numpy as np
from timestral import TimestralForecaster

P10, P90 = 0.1, 0.9      # the pair every example prints as its "80% band"


@dataclass(frozen=True)
class Forecast:
    """Quantiles, always shaped (variates, horizon)."""

    median: np.ndarray
    low: np.ndarray
    high: np.ndarray


def predict(context, horizon: int, model: str) -> Forecast:
    """Forecast `horizon` steps past `context`.

    `context` is one variate as a 1-D sequence of T values, or several variates
    as a 2-D (variates, T) sequence — the model then forecasts them jointly.
    Returned arrays are always 2-D, so a single variate is row 0.
    """
    values = np.asarray(context, dtype=np.float32)
    if values.ndim == 1:
        model_input = values                    # B=1, V=1
    elif values.ndim == 2:
        model_input = values[None, :, :]         # B=1, V variates, forecast jointly
    else:
        raise ValueError(f"context must be 1-D or 2-D, got {values.ndim}-D")

    forecast = TimestralForecaster.from_pretrained(model).forecast(
        model_input, prediction_length=horizon
    )
    return Forecast(
        median=forecast.median.numpy().reshape(-1, horizon),
        low=forecast.quantile(P10).numpy().reshape(-1, horizon),
        high=forecast.quantile(P90).numpy().reshape(-1, horizon),
    )
