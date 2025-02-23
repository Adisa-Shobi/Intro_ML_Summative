#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Inflation rates
inflation_rates = {
    "2021": 18.17,
    "2022": 15.70,
    "2023": 24.66,
    "2024": 34.80
}


def adjust_for_inflation(base_price):
    """Adjust price for inflation from 2021 to 2024"""
    price = base_price
    for year, rate in inflation_rates.items():
        price *= (1 + rate/100)
    return round(price, 2)


# Real baseline prices from Nigerian markets [morning, afternoon, evening]
real_prices = {
    "Agric eggs medium size": [adjust_for_inflation(price) for price in [524.47, 654.90, 668.96]],
    "Beans brown": [adjust_for_inflation(price) for price in [368.98, 509.65, 530.10]],
    "Beans white black eye": [adjust_for_inflation(price) for price in [345.03, 497.54, 505.94]],
    "Beef Bone in": [adjust_for_inflation(price) for price in [1128.10, 1440.70, 1496.47]],
    "Beef boneless": [adjust_for_inflation(price) for price in [1456.03, 1922.22, 1955.90]],
    "Catfish (obokun) fresh": [adjust_for_inflation(price) for price in [1084.84, 1330.07, 1341.04]],
    "Catfish dried": [adjust_for_inflation(price) for price in [1723.52, 2050.31, 2090.01]],
    "Chicken Wings": [adjust_for_inflation(price) for price in [992.30, 1222.77, 1238.00]],
    "Tomato": [adjust_for_inflation(price) for price in [267.45, 393.08, 409.96]],
    "Onion bulb": [adjust_for_inflation(price) for price in [285.23, 378.26, 378.59]],
    "Sweet potato": [adjust_for_inflation(price) for price in [152.53, 227.23, 228.05]],
    "Yam tuber": [adjust_for_inflation(price) for price in [244.82, 339.76, 353.56]]
}

# Storage types for each item
storage_types = {
    "Agric eggs medium size": "room",
    "Beans brown": "room",
    "Beans white black eye": "room",
    "Beef Bone in": "cold",
    "Beef boneless": "cold",
    "Catfish (obokun) fresh": "cold",
    "Catfish dried": "room",
    "Chicken Wings": "cold",
    "Tomato": "room",
    "Onion bulb": "room",
    "Sweet potato": "room",
    "Yam tuber": "room"
}

# Selected months with seasonal patterns
selected_months = {
    "January": {"wetSeason": 0, "inSeason": ["Tomato", "Sweet potato"]},
    "April": {"wetSeason": 1, "inSeason": ["Onion bulb", "Yam tuber"]},
    "July": {"wetSeason": 1, "inSeason": ["Sweet potato", "Yam tuber"]},
    "October": {"wetSeason": 0, "inSeason": ["Tomato", "Onion bulb"]}
}


def generate_price_and_demand(baseline_prices, time_of_day, is_in_season, is_wet_season, days_since_delivery, storage, item):
    """Generate price and demand with real-world complexities"""
    time_index = {"Morning": 0, "Afternoon": 1, "Evening": 2}[time_of_day]
    base_price = baseline_prices[time_index]

    # Market events
    is_market_day = np.random.random() < 0.15
    is_religious_period = np.random.random() < 0.12

    # Price calculation with real-world factors
    price = base_price

    # Time of day with noise
    time_multiplier = {
        "Morning": np.random.normal(1.3, 0.1),
        "Afternoon": np.random.normal(1.0, 0.08),
        "Evening": np.random.normal(0.7, 0.15)
    }[time_of_day]

    price *= time_multiplier

    # Seasonal effects with noise
    if is_in_season:
        seasonal_discount = np.random.normal(0.7, 0.1)
        price *= max(seasonal_discount, 0.5)

    # Weather impact
    if is_wet_season:
        if storage == "room":
            weather_impact = np.random.normal(1.25, 0.15)
            price *= weather_impact

            if item in ["Tomato", "Onion bulb", "Sweet potato"]:
                price *= np.random.normal(1.15, 0.2)

    # Storage and delivery effects
    if storage == "room":
        deterioration = 1 + (days_since_delivery *
                             np.random.normal(0.05, 0.02))
        price *= deterioration
    else:
        price *= np.random.normal(1.3, 0.05)

    # Event effects
    if is_market_day:
        price *= np.random.normal(0.9, 0.05)
    if is_religious_period:
        price *= np.random.normal(1.2, 0.1)

    # General market noise
    price *= np.random.normal(1.0, 0.03)
    price = round(max(price, base_price * 0.4), 2)

    # Sellout probability calculation
    base_sellout_prob = {
        "Morning": 0.65 + np.random.normal(0, 0.1),
        "Afternoon": 0.45 + np.random.normal(0, 0.08),
        "Evening": 0.25 + np.random.normal(0, 0.15)
    }[time_of_day]

    # Probability adjustments
    if is_in_season:
        base_sellout_prob += np.random.normal(0.2, 0.05)
    if price < base_price * 0.8:
        base_sellout_prob += np.random.normal(0.25, 0.08)
    if is_wet_season and storage == "room":
        base_sellout_prob -= np.random.normal(0.15, 0.05)
    if is_market_day:
        base_sellout_prob += np.random.normal(0.2, 0.05)
    if is_religious_period:
        base_sellout_prob += np.random.normal(0.15, 0.07)
    if item in ["Tomato", "Onion bulb"]:
        base_sellout_prob += np.random.normal(0.1, 0.03)

    # Random market behavior
    if np.random.random() < 0.02:
        base_sellout_prob = np.random.random()

    sellout_prob = max(min(base_sellout_prob, 1), 0)
    sold_out = np.random.random() < sellout_prob

    return price, sold_out, is_market_day, is_religious_period


def generate_dataset():
    """Generate dataset with real-world complexities"""
    data = []
    time_slots = ["Morning", "Afternoon", "Evening"]
    days_per_month = 15

    stats = {
        "total": 0,
        "wet_season": 0,
        "cold_storage": 0,
        "in_season": 0,
        "sold_out": 0,
        "market_days": 0,
        "religious_days": 0
    }

    for month, season_info in selected_months.items():
        for day in range(1, days_per_month + 1):
            transport_disruption = np.random.random() < 0.05

            for time_slot in time_slots:
                if np.random.random() < {
                    "Morning": np.random.normal(0.96, 0.02),
                    "Afternoon": np.random.normal(0.94, 0.02),
                    "Evening": np.random.normal(0.92, 0.03)
                }[time_slot]:

                    for item, prices in real_prices.items():
                        is_in_season = item in season_info["inSeason"]
                        storage = storage_types[item]
                        days_since_delivery = np.random.randint(1, 6)

                        price, sold_out, is_market_day, is_religious_period = generate_price_and_demand(
                            prices,
                            time_slot,
                            is_in_season,
                            season_info["wetSeason"],
                            days_since_delivery,
                            storage,
                            item
                        )

                        # Quantity calculation
                        base_quantity = np.random.normal(50, 15)

                        if time_slot == "Morning":
                            base_quantity *= np.random.normal(1.2, 0.1)
                        elif time_slot == "Evening":
                            base_quantity *= np.random.normal(0.8, 0.15)

                        if transport_disruption:
                            base_quantity *= np.random.normal(0.7, 0.2)

                        if is_market_day:
                            base_quantity *= np.random.normal(1.3, 0.1)

                        quantity = int(max(round(base_quantity), 10))

                        # Update statistics
                        stats["total"] += 1
                        if season_info["wetSeason"]:
                            stats["wet_season"] += 1
                        if storage == "cold":
                            stats["cold_storage"] += 1
                        if is_in_season:
                            stats["in_season"] += 1
                        if sold_out:
                            stats["sold_out"] += 1
                        if is_market_day:
                            stats["market_days"] += 1
                        if is_religious_period:
                            stats["religious_days"] += 1

                        data.append({
                            "Date": f"2024-{month}-{day:02d}",
                            "Month": month,
                            "TimeOfDay": time_slot,
                            "Item": item,
                            "PriceNaira": price,
                            "Quantity": quantity,
                            "IsWetSeason": season_info["wetSeason"],
                            "IsInSeason": int(is_in_season),
                            "DaysSinceDelivery": days_since_delivery,
                            "StorageType": storage,
                            "CurrentInflationRate": inflation_rates["2024"],
                            "IsMarketDay": int(is_market_day),
                            "IsReligiousPeriod": int(is_religious_period),
                            "TransportDisruption": int(transport_disruption),
                            "SoldOut": int(sold_out)
                        })

    df = pd.DataFrame(data)

    print("\nDataset Statistics:")
    print(f"Total Records: {stats['total']}")
    print(f"Wet Season: {stats['wet_season']/stats['total']*100:.1f}%")
    print(f"Cold Storage: {stats['cold_storage']/stats['total']*100:.1f}%")
    print(f"In Season: {stats['in_season']/stats['total']*100:.1f}%")
    print(f"Sold Out: {stats['sold_out']/stats['total']*100:.1f}%")
    print(f"Market Days: {stats['market_days']/stats['total']*100:.1f}%")
    print(f"Religious Days: {stats['religious_days']/stats['total']*100:.1f}%")

    return df


# Generate and save the dataset
if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv('nigerian_market_prices_2024.csv', index=False)

    print("\nSample of the dataset:")
    print(df.head())

    # Select only numeric columns for correlation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_columns].corr(
    )['SoldOut'].sort_values(ascending=False)
    print("\nCorrelation with SoldOut:")
    print(correlations)

    # Optional: Print some additional insights
    print("\nFeature Distributions:")
    print(df[numeric_columns].describe())
