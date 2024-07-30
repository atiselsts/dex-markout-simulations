#
# This file simulates a constant product AMM DEX
#

from math import sqrt

############################################################

# Constants for the examples

ETH_PRICE = 3000

POOL_LIQUIDITY_USD = 1_000_000_000

# LP fee, in parts per million (pips)
POOL_FEE_PIPS = 500 # corresponds to 0.05%

# For simplicity, assume no gas fees
DEFAULT_BASEFEE_USD = 0

############################################################

class DEX:
    def __init__(self, pool_liquidity_usd=POOL_LIQUIDITY_USD):

        POOL_RESERVES_USD = POOL_LIQUIDITY_USD / 2
        POOL_RESERVES_ETH = POOL_RESERVES_USD / ETH_PRICE

        # -- parameters
        self.fee_pips = POOL_FEE_PIPS
        self.fee_factor = 1_000_000 / (1_000_000 - self.fee_pips)
        self.basefee_usd = DEFAULT_BASEFEE_USD
        self.block_time_sec = 12
        # -- pool's state
        # the price is fully determined by the reserves (real or virtual)
        self.reserve_x = POOL_RESERVES_ETH
        self.reserve_y = POOL_RESERVES_USD
        # -- cumulative metrics
        self.volume = 0
        self.volume_arb = 0
        self.lp_fees = 0
        self.lvr = 0
        self.sbp_profits = 0
        self.basefees = 0
        self.num_tx = 0
        # cumulative markouts (to CEX prices). The keys are minutes. "0" is for after-swap markouts
        self.markouts = {0: [], 1: [], 3: [], 10: [], 30: []}
        # debugging
        self.debug_log = False
        self.preset_target_price = None


    def minutes_to_blocks(self, time_minutes):
        seconds = 60 * time_minutes
        return seconds // self.block_time_sec


    def set_fee_bps(self, fee_bps):
        self.fee_pips = fee_bps * 100
        self.fee_factor = 1_000_000 / (1_000_000 - self.fee_pips)


    def set_basefee_usd(self, basefee_usd):
        self.basefee_usd = basefee_usd


    def price(self):
        return self.reserve_y / self.reserve_x


    def liquidity(self):
        return sqrt(self.reserve_x * self.reserve_y)

    
    def get_amounts_to_target_price(self, target_price):
        if self.preset_target_price is not None:
            target_price = self.preset_target_price

        sqrt_target_price = sqrt(target_price)
        L = self.liquidity()
        delta_x = L / sqrt_target_price - self.reserve_x
        delta_y = L * sqrt_target_price - self.reserve_y
        return (delta_x, delta_y)


    def swap_x_to_y(self, amount_in_x, cex_prices, index):
        amount_in_x_without_fee = amount_in_x / self.fee_factor

        assert amount_in_x > amount_in_x_without_fee
        price = self.price()
        self.lp_fees += (amount_in_x - amount_in_x_without_fee) * price
        self.reserve_x += amount_in_x_without_fee
        y_out = amount_in_x_without_fee * self.reserve_y / self.reserve_x
        self.reserve_y -= y_out

        self.volume += amount_in_x * price
        self.num_tx += 1
        self.basefees += self.basefee_usd
        if self.debug_log:
            print("swap x to y", y_out)
        self.add_markouts(amount_in_x, -y_out, cex_prices, index)
        return y_out


    def swap_y_to_x(self, amount_in_y, cex_prices, index):
        amount_in_y_without_fee = amount_in_y / self.fee_factor

        assert amount_in_y > amount_in_y_without_fee
        self.lp_fees += amount_in_y - amount_in_y_without_fee
        self.reserve_y += amount_in_y_without_fee
        x_out = amount_in_y_without_fee * self.reserve_x / self.reserve_y
        self.reserve_x -= x_out

        self.volume += amount_in_y
        self.num_tx += 1
        self.basefees += self.basefee_usd
        if self.debug_log:
            print("swap y to x", amount_in_y)
        self.add_markouts(-x_out, amount_in_y, cex_prices, index)
        return x_out


    def get_target_price(self, cex_price):
        dex_price = self.price()
        if cex_price > dex_price:
            target_price = cex_price / self.fee_factor
            if target_price < dex_price:
                return None
        else:
            target_price = cex_price * self.fee_factor
            if target_price > dex_price:
                return None
        return target_price

    
    def get_non_arbitrage_region(self):
        p = self.price()
        # this is accurate as long as the gas fee for swaps is zero
        return [p / self.fee_factor, p * self.fee_factor]


    def add_zero_markouts(self):
        for markout_minutes in self.markouts.keys():
            self.markouts[markout_minutes].append(0.0)

    def maybe_arbitrage(self, cex_prices, index, include_markouts):
        cex_price = cex_prices[index]
        target_price = self.get_target_price(cex_price)
        #print(cex_price, target_price)
        if target_price is None:
            # the trade does not happen because the CEX/DEX price difference is below the LP fee
            if include_markouts:
                self.add_zero_markouts()
            return False

        delta_x, delta_y = self.get_amounts_to_target_price(target_price)
        # compute the LP fees using CEX prices
        # the assumption here is that LPs do not accumulate or compound their fees, but withdraw and rapidly convert to USD
        if delta_x > 0:
            delta_x_with_fee = delta_x * self.fee_factor
            delta_y_with_fee = delta_y
            lp_fee = (delta_x_with_fee - delta_x) * cex_price
        else:
            delta_x_with_fee = delta_x
            delta_y_with_fee = delta_y * self.fee_factor
            lp_fee = delta_y_with_fee - delta_y

        single_transaction_lvr = -(delta_x * cex_price + delta_y)
        sbp_profit = single_transaction_lvr - lp_fee - self.basefee_usd
        if sbp_profit <= 0.0:
            # the trade does not happen due to the friction from the blockchain base fee 
            if self.debug_log:
                print("sbp_profit <= 0.0:", single_transaction_lvr, lp_fee, sbp_profit)
            if include_markouts:
                self.add_zero_markouts()
            return False

        # trade happens; first update the pool's state
        if self.debug_log:
            new_reserve_x = self.reserve_x + delta_x
            new_reserve_y = self.reserve_y + delta_y
            lp_loss_vs_lvr = (single_transaction_lvr - lp_fee) / single_transaction_lvr
            print(f" DEX price: {self.reserve_y/self.reserve_x:.4f}->{new_reserve_y/new_reserve_x:.4f} CEX price: {cex_price:.4f} LP fee={lp_fee:.2f} LVR={single_transaction_lvr:.2f} loss: {100*lp_loss_vs_lvr:.1f}%")

        self.reserve_x += delta_x
        self.reserve_y += delta_y

        # then update the cumulative metrics
        trade_volume = abs(delta_y_with_fee)
        self.volume += trade_volume
        self.volume_arb += trade_volume

        self.lp_fees += lp_fee
        self.lvr += single_transaction_lvr
        self.sbp_profits += sbp_profit
        self.basefees += self.basefee_usd
        self.num_tx += 1
        if include_markouts:
            self.add_markouts(delta_x_with_fee, delta_y_with_fee, cex_prices, index)

        return True

    def add_markouts(self, delta_x, delta_y, cex_prices, index):
        cex_price = cex_prices[index]
        execution_price = abs(delta_y / delta_x)
        for markout_minutes in self.markouts.keys():
            num_blocks = self.minutes_to_blocks(markout_minutes)
            future_index = index + num_blocks
            if future_index < len(cex_prices):
                future_price = cex_prices[future_index]
                markout = delta_x * (future_price - execution_price)
                if self.debug_log:
                    print(f"  {markout_minutes}: {markout:.6f} ({execution_price} vs {future_price})")
                self.markouts[markout_minutes].append(markout)
            else:
                self.markouts[markout_minutes].append(0.0)
