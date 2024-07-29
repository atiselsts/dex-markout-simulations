#!/usr/bin/env python

#
# This script simulates arbitrage-only trading
#

import matplotlib.pyplot as pl
import numpy as np
from ing_theme_matplotlib import mpl_style
from dex import DEX, ETH_PRICE, POOL_LIQUIDITY_USD
# Constants for plotting
pl.rcParams["savefig.dpi"] = 200

# the volatility of ETH price per one year - approximately matches the recent year's data
ETH_VOLATILITY = 0.5

ETH_VOLATILITY_PER_SECOND = ETH_VOLATILITY / np.sqrt(365 * 24 * 60 * 60)

SIMULATION_DURATION_SEC = 86400

NUM_SIMULATIONS = 50

BLOCK_TIME = 12

############################################################

def get_price_paths(n, sigma, mu, M=NUM_SIMULATIONS):
    St = np.exp((mu - sigma ** 2 / 2) + sigma * np.random.normal(0, 1, size=(M, n-1)).T)

    # we want the initial prices to be randomly distributed in the pool's non-arbitrage space
    price_low, price_high = DEX().get_non_arbitrage_region()
    initial_prices = np.random.uniform(price_low / ETH_PRICE, price_high / ETH_PRICE, M)
    St = np.vstack([initial_prices, St])

    St = ETH_PRICE * St.cumprod(axis=0)
    return St

############################################################

def estimate_performance(prices, noise_trades=None, swap_fee_bps=None, basefee_usd=None, include_arb=True):
    dex = DEX()
    #dex.debug_log = True
    dex.block_time_sec = BLOCK_TIME
    if swap_fee_bps is not None:
        dex.set_fee_bps(swap_fee_bps)
    if basefee_usd is not None:
        dex.set_basefee_usd(basefee_usd)
    lp_pnl = []
    if noise_trades is None:
        for i in range(len(prices)):
            dex.maybe_arbitrage(prices, i, include_arb)
            lp_pnl.append(dex.lp_fees - dex.lvr)
    else:
        for i in range(len(prices)):
            # first execute the arbitrage (may include a backrun)
            dex.maybe_arbitrage(prices, i, include_arb)
            # then execute the noise trade
            cex_price = prices[i]
            trade_amount = noise_trades[i]
            if trade_amount < 0:
                trade_amount = -trade_amount / cex_price
                dex.swap_x_to_y(trade_amount, prices, i)
                lp_pnl.append(dex.lp_fees - dex.lvr)
            elif trade_amount > 0:
                dex.swap_y_to_x(trade_amount, prices, i)
                lp_pnl.append(dex.lp_fees - dex.lvr)

    return dex.lvr, dex.lp_fees, dex.volume, dex.markouts, lp_pnl


############################################################

# for simpler-to-read plots remove most samples from the signal
def decimate(x, n=100):
    result = [0] * ((len(x) + n - 1) // n)
    for i in range(0, len(x), n):
        result[i//n] = x[i]
    return result

############################################################

def plot_performance_arb_only(all_prices, duration_seconds, swap_fee_bps):
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3.5))

    num_blocks = int(duration_seconds // BLOCK_TIME)

    all_volume = []
    all_markouts_0 = []
    all_markouts_1 = []
    all_markouts_3 = []
    all_markouts_10 = []
    all_markouts_30 = []
    all_lp_pnl = []

    for sim in range(all_prices.shape[1]):
        prices = all_prices[:,sim]
        if num_blocks is not None:
            prices = prices[:num_blocks]
        lvr, lp_fees, volume, markouts, lp_pnl = \
            estimate_performance(prices, None, swap_fee_bps)
        all_volume.append(volume)
        all_markouts_0.append(markouts[0])
        all_markouts_1.append(markouts[1])
        all_markouts_3.append(markouts[3])
        all_markouts_10.append(markouts[10])
        all_markouts_30.append(markouts[30])
        all_lp_pnl.append(lp_pnl)

    x = np.linspace(0, duration_seconds / 60, num_blocks)
    x = decimate(x)
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_0,axis=0))), label="Immediate", marker="D", color="black")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_1,axis=0))), label="1 min", marker="d", color="brown")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_3,axis=0))), label="3 min", marker="p", color="red")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_10,axis=0))), label="10 min", marker="s", color="orange")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_30,axis=0))), label="30 min", marker="^", color="green")

    pl.plot(x, decimate(np.mean(all_lp_pnl, axis=0)), label="LP Pnl (using LVR)", marker="x", color="grey")

    pl.xlabel("Minutes")
    pl.ylabel("Cumulative markouts, $")
    pl.legend()
    #pl.ylim(-100, +10)
    pl.title(f"Markouts for {swap_fee_bps} bps pool, arbitrage only")

    avg_volume = np.mean(volume)
    print(f"arb only: volume={avg_volume / 1e6:.2f} M")

    pl.savefig(f"markouts_arb_only_{swap_fee_bps}bps.png", bbox_inches='tight')

############################################################

def generate_trades(num_blocks, expected_volume_per_block):
    if False:
        # generates exponentially distributed trades
        values = np.random.exponential(scale=1.0, size=num_blocks)
        values *= expected_volume_per_block
    else:
        # generates Pareto distributed trades
        s = num_blocks * expected_volume_per_block
        alpha = 1.5
        values = np.random.pareto(alpha, num_blocks)
        # Scale the values to match the desired sum
        values *= s / values.sum()

    # print(expected_volume_per_block, np.mean(values))
    # print(values)

    # make half the results negative (determines trade direction)
    indices = np.random.permutation(num_blocks)
    num_to_invert = num_blocks // 2
    values[indices[:num_to_invert]] *= -1
    return values

# try to match the noise trades and normal trades 50:50
def get_expected_volume_per_block(swap_fee_bps):
    return ({
        5: 12e6,
        30: 1.8e6,
        100: 0.01e6
    }[swap_fee_bps] / 7200) / 5

############################################################

def plot_performance_both(all_prices, duration_seconds, swap_fee_bps):
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3.5))

    num_blocks = int(duration_seconds // BLOCK_TIME)

    all_markouts_0 = []
    all_markouts_1 = []
    all_markouts_3 = []
    all_markouts_10 = []
    all_markouts_30 = []
    all_volume = []
    all_lp_pnl = []

    noise_trades = generate_trades(num_blocks, get_expected_volume_per_block(swap_fee_bps))

    for sim in range(all_prices.shape[1]):
        prices = all_prices[:,sim]
        if num_blocks is not None:
            prices = prices[:num_blocks]
        lvr, lp_fees, volume, markouts, lp_pnl = \
            estimate_performance(prices, noise_trades, swap_fee_bps)
        all_volume.append(volume)
        all_markouts_0.append(markouts[0])
        all_markouts_1.append(markouts[1])
        all_markouts_3.append(markouts[3])
        all_markouts_10.append(markouts[10])
        all_markouts_30.append(markouts[30])
        all_lp_pnl.append(lp_pnl)

    x = np.linspace(0, duration_seconds / 60, num_blocks)
    x = decimate(x)
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_0,axis=0)), 200), label="Immediate", marker="D", color="black")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_1,axis=0)), 200), label="1 min", marker="d", color="brown")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_3,axis=0)), 200), label="3 min", marker="p", color="red")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_10,axis=0)), 200), label="10 min", marker="s", color="orange")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_30,axis=0)), 200), label="30 min", marker="^", color="green")
    pl.plot(x, decimate(np.mean(all_lp_pnl, axis=0)), label="LP Pnl (using LVR)", marker="x", color="grey")

    pl.xlabel("Minutes")
    pl.ylabel("Cumulative markouts, $")
    pl.legend()
    pl.title(f"Markouts for {swap_fee_bps} bps pool, arbitrage & noise traders")

    avg_volume = np.mean(volume)
    print(f"arb and noise: volume={avg_volume / 1e6:.3f} M")

    pl.savefig(f"markouts_both_{swap_fee_bps}bps.png", bbox_inches='tight')

############################################################

def plot_performance_noise_only(all_prices, duration_seconds, swap_fee_bps):
    fig, ax = pl.subplots()
    fig.set_size_inches((5, 3.5))

    num_blocks = int(duration_seconds // BLOCK_TIME)

    all_markouts_0 = []
    all_markouts_1 = []
    all_markouts_3 = []
    all_markouts_10 = []
    all_markouts_30 = []
    all_volume = []
    all_lp_pnl = []

    noise_trades = generate_trades(num_blocks, get_expected_volume_per_block(swap_fee_bps))

    for sim in range(all_prices.shape[1]):
        prices = all_prices[:,sim]
        if num_blocks is not None:
            prices = prices[:num_blocks]
        lvr, lp_fees, volume, markouts, lp_pnl = \
            estimate_performance(prices, noise_trades, swap_fee_bps, include_arb=False)
        all_volume.append(volume)
        all_markouts_0.append(markouts[0])
        all_markouts_1.append(markouts[1])
        all_markouts_3.append(markouts[3])
        all_markouts_10.append(markouts[10])
        all_markouts_30.append(markouts[30])
        all_lp_pnl.append(lp_pnl)

    x = np.linspace(0, duration_seconds / 60, num_blocks)
    x = decimate(x)
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_0,axis=0))), label="Immediate", marker="D", color="black")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_1,axis=0))), label="1 min", marker="d", color="brown")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_3,axis=0))), label="3 min", marker="p", color="red")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_10,axis=0))), label="10 min", marker="s", color="orange")
    pl.plot(x, decimate(np.cumsum(np.mean(all_markouts_30,axis=0))), label="30 min", marker="^", color="green")

    pl.xlabel("Minutes")
    pl.ylabel("Cumulative markouts, $")
    pl.legend()
    pl.title(f"Markouts for {swap_fee_bps} bps pool, noise only")

    avg_volume = np.mean(volume)
    print(f"arb and noise: volume={avg_volume / 1e6:.3f} M")

    pl.savefig(f"markouts_noise_only_{swap_fee_bps}bps.png", bbox_inches='tight')

############################################################

def simulate_some_blocks():
    n = SIMULATION_DURATION_SEC
    all_prices = get_price_paths(n, sigma=ETH_VOLATILITY_PER_SECOND, mu=0.0)
    for swap_fee_bps in [5, 30]:
    #for swap_fee_bps in [5]:
        plot_performance_arb_only(all_prices, SIMULATION_DURATION_SEC - 2000, swap_fee_bps)
        plot_performance_both(all_prices, SIMULATION_DURATION_SEC - 2000, swap_fee_bps)
        plot_performance_noise_only(all_prices, SIMULATION_DURATION_SEC - 2000, swap_fee_bps)

############################################################x
    
def main():
    mpl_style(False)
    np.random.seed(12345)
    simulate_some_blocks()


if __name__ == '__main__':
    main()
    print("all done!")
