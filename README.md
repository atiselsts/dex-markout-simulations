# Code for DEX markout simulations

This repository contains Python code for the markout measurements on simulated price trajectories.

# Assumptions

It assumes:
* A DEX with deep liquidity of $1 billion. This is large, but not unrealistic,
  as the ETH/USDC 0.05% pool on Uniswap v3 mainnet has a comparable *virtual* liquidity depth.
* The DEX has a `xy=k` pool with a stable / volatile asset pair.
* The volatility of the volatile asset is either 50% per year, or 5% per day (~90% per year)
  depending on the simulation. This matches the performance of ETH is less or more volatile periods.
* The base fee of arbitrage transactions is constant and does not depend on the price action.
* The swap fees are not compounding.

It also makes the standard assumptions behing the LVR model.
* There is a CEX which trades the volatile asset.
* The traders are not required to pay any trading fees on the CEX.
* The liquidity on the CEX is infinitely deep.
* There is a CEX/DEX arbitrager that has unlimited amount of stable assets, fast connections
  to both CEX and DEX, and will take all profitable trades at their maximum volume.
  
For simulations that include noise traders, additionally:
* the size of the trades is distibuted according to Pareto distribution

# Contents

* `simulate.py` - simulation scenarios
* `dex.py` - core DEX implementation

# Simulations

The current implementation of price path simulations has significant RAM usage
and may require several minutes to complete for a large number of simulations.
