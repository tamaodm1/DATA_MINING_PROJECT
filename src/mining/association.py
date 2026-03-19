"""
association.py – Khai phá luật kết hợp với Apriori/FP-Growth.
Tìm điều kiện thời tiết đồng xuất hiện (theo mùa).
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


class AssociationMiner:
    def __init__(self, cfg: dict):
        mc = cfg["mining"]["association"]
        self.min_support = mc["min_support"]
        self.min_confidence = mc["min_confidence"]
        self.min_lift = mc["min_lift"]

    def mine(self, basket_df: pd.DataFrame, algorithm: str = "fpgrowth") -> pd.DataFrame:
        """
        Chạy Apriori hoặc FP-Growth trên basket one-hot.
        Trả về DataFrame các frequent itemsets.
        """
        print(f"[association] Mining with {algorithm}, support={self.min_support}")
        if algorithm == "apriori":
            freq_items = apriori(basket_df, min_support=self.min_support,
                                  use_colnames=True, verbose=0)
        else:
            freq_items = fpgrowth(basket_df, min_support=self.min_support,
                                   use_colnames=True, verbose=0)
        print(f"[association] Found {len(freq_items)} frequent itemsets")
        return freq_items

    def get_rules(self, freq_items: pd.DataFrame) -> pd.DataFrame:
        """Tạo luật kết hợp từ frequent itemsets."""
        rules = association_rules(
            freq_items, metric="lift", min_threshold=self.min_lift
        )
        rules = rules[rules["confidence"] >= self.min_confidence]
        rules = rules.sort_values("lift", ascending=False)
        print(f"[association] Generated {len(rules)} rules after filtering")
        return rules

    def top_rules(self, rules: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Trả về top-N luật theo lift."""
        return rules.head(n)[["antecedents", "consequents", "support",
                               "confidence", "lift"]].reset_index(drop=True)

    def mine_by_season(self, df: pd.DataFrame, basket_fn) -> dict:
        """
        Chạy association mining theo từng mùa.
        basket_fn: hàm nhận df và trả về basket_df.
        """
        results = {}
        for season in df["Season"].unique():
            sub = df[df["Season"] == season]
            basket = basket_fn(sub)
            freq = self.mine(basket)
            if len(freq) > 0:
                rules = self.get_rules(freq)
                results[season] = rules
                print(f"  Season={season}: {len(rules)} rules")
        return results
