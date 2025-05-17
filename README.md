
# 🧠 Reinforcement Learning for Nifty Call Options 📈

This project implements a complete pipeline to simulate, train, evaluate, and backtest a **Reinforcement Learning agent** for trading **Nifty 50 Call Options** using PPO (Proximal Policy Optimization). It includes:

- Simulated ATM CE/PE price generation from Nifty spot data
- Technical and financial feature engineering
- Black-Scholes Greeks calculation
- A custom OpenAI Gym trading environment
- PPO agent training using Stable Baselines3
- Backtest and performance analysis using VectorBT

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `new_test.ipynb` | Full pipeline notebook: simulation → training → evaluation |
| `nifty_hourly_data_2020_to_today.csv` | Raw hourly Nifty spot price data |
| `simulated_nifty_options.csv` | Generated CE/PE options data |
| `nifty_rl_ready.csv` | Final training-ready dataset with features |
| `ppo_ce_actions.csv` | Agent’s trading actions |
| `ppo_simple_ce.zip` | Trained PPO model |
| `SimpleNiftyCallEnv` | Custom trading environment for RL agent |

---

## ⚙️ Simulation Logic

Options are priced using:
- Intrinsic value from spot-close price
- Linear **theta decay** for time value
- **IV noise** via Gaussian perturbation
- Expiry logic and behavioral premiums
- Exported hourly CE/PE prices to `simulated_nifty_options.csv`

---

## 📈 Feature Engineering

Features used for training:
- Technical: `SMA`, `RSI`, `Momentum`
- Black-Scholes: `Delta`, `Vega`, `Theta` for CE/PE
- Time: `hour`, `day_of_week`, `minutes_into_day`
- Lag features: previous 3 timesteps of spot, CE, PE prices

---

## 🎮 Custom Environment: `SimpleNiftyCallEnv`

- Actions: `0=Hold`, `1=Buy CE`, `2=Sell CE`
- Reward:
  - +5 for successful buy
  - Realized PnL - trading costs for sell
- Observation: current market state + position flag + normalized PnL
- Transaction costs include brokerage, STT, SEBI, GST

---

## 🤖 PPO Model Training

- Framework: `Stable-Baselines3`
- Agent: `MlpPolicy` PPO
- Training Steps: `100,000`
- Evaluation every `10,000` steps
- Best model checkpointed

---

## 📊 Backtest Performance Summary (VectorBT)

| Metric | Value |
|--------|-------|
| **Total Return** | **+400.24%** |
| **Final Portfolio Value** | ₹500,244.86 |
| **Max Drawdown** | 1.40% |
| **Sharpe Ratio** | 90.54 |
| **Sortino Ratio** | 223.20 |
| **Win Rate** | 95.8% |
| **Profit Factor** | 88.3 |
| **Avg Trade Duration** | ~6 min |
| **Best Trade** | 369,060% |
| **Worst Trade** | -99.77% |

> ⚠️ **Note:** Outliers in CE prices can cause abnormal returns — price clipping or normalization is recommended before production use.

---

## 📦 Requirements

```bash
pip install pandas numpy gym stable-baselines3 vectorbt
```

---

## 🧪 How to Run

1. Prepare hourly Nifty spot data in `nifty_hourly_data_2020_to_today.csv`
2. Run the notebook: this generates simulated options, engineered dataset, trains PPO, evaluates and backtests
3. Review final logs and VectorBT plots

---

## 📌 Future Enhancements

- Add position management (multi-lot)
- Extend to PE or straddle/strangle trading
- Integrate with live NSE/Kite API
- Implement risk-based reward shaping

---

## 📜 License

MIT License

---

## 🤝 Acknowledgements

Built using:
- [`Stable-Baselines3`](https://github.com/DLR-RM/stable-baselines3)
- [`VectorBT`](https://github.com/polakowo/vectorbt)
- [`Gym`](https://www.gymlibrary.dev/)
