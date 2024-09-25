from numpy import exp, sqrt, log
from scipy.stats import norm


class BlackScholes:
    def __init__(
        self,
        time_to_expiry: float,
        strike_price: float,
        spot_price: float,
        volatility: float,
        risk_free_rate: float,
    ):
        self.time_to_expiry = time_to_expiry
        self.strike_price = strike_price
        self.spot_price = spot_price
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate

    def calculate_prices_and_greeks(self):
        t = self.time_to_expiry
        K = self.strike_price
        S = self.spot_price
        sigma = self.volatility
        r = self.risk_free_rate

        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)

        self.call_price = S * norm.cdf(d1) - K * exp(-r * t) * norm.cdf(d2)
        self.put_price = K * exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

        # Greeks
        self.call_delta = norm.cdf(d1)
        self.put_delta = -norm.cdf(-d1)

        self.gamma = norm.pdf(d1) / (S * sigma * sqrt(t))
        self.call_gamma = self.gamma
        self.put_gamma = self.gamma


if __name__ == "__main__":
    time_to_expiry = 2
    strike_price = 90
    spot_price = 100
    volatility = 0.2
    risk_free_rate = 0.05

    # Black Scholes
    bs_model = BlackScholes(
        time_to_expiry=time_to_expiry,
        strike_price=strike_price,
        spot_price=spot_price,
        volatility=volatility,
        risk_free_rate=risk_free_rate)
    bs_model.calculate_prices_and_greeks()