import re

class LegalMetrologyEngine:
    def __init__(self):
        # Rule 32: Penalty slabs
        self.PENALTY_MINOR = 2000
        self.PENALTY_MAJOR = 4000
        self.total_penalty = 0
        self.violations = []

    def log_violation(self, rule, description, penalty):
        self.violations.append({"rule": rule, "desc": description})
        self.total_penalty += penalty

    def check_rule_6_1_c_net_quantity(self, qty_text):
        """Rule 6(1)(c) & Rule 13: Strict SI unit check."""
        qty_text = qty_text.lower()
        
        # Check for illegal terms
        illegal_terms = ['gm', 'gms', 'dozen', 'gross']
        for term in illegal_terms:
            if term in qty_text.split():
                self.log_violation(
                    rule="Rule 6(1)(c) / Rule 13",
                    description=f"Illegal unit/term used: '{term}'. Must use SI units (g, kg, ml, l).",
                    penalty=self.PENALTY_MINOR
                )
                return False
                
        # Valid SI check (basic regex for Number + valid unit)
        if not re.search(r'\d+\s*(g|kg|ml|l|m|cm|n|u)$', qty_text):
             self.log_violation(
                rule="Rule 6(1)(c)",
                description=f"Invalid or missing SI unit in: '{qty_text}'",
                penalty=self.PENALTY_MAJOR
             )
             return False
        return True

    def check_rule_6_1_e_mrp(self, mrp_text):
        """Rule 6(1)(e) & Rule 18: Statutory MRP formatting."""
        # Must contain exact phrasing or close accepted variations
        pattern = r'MRP\s*(Rs\.?|₹)\s*\d+(\.\d{2})?\s*\(?(incl\.?\s*of\s*all\s*taxes)\)?'
        
        if not re.search(pattern, mrp_text, re.IGNORECASE):
            self.log_violation(
                rule="Rule 6(1)(e) / Rule 18",
                description="MRP format invalid. Must follow 'MRP Rs. XX.XX incl. of all taxes'.",
                penalty=self.PENALTY_MAJOR
            )
            return False
        return True

    def check_rule_7_font_height(self, box_height_px, image_dpi, panel_area_cm2):
        """Rule 7 & Tables I & II: Min font height based on Principal Display Panel area."""
        # Convert pixel height to mm
        height_mm = (box_height_px * 25.4) / image_dpi
        
        # Table I logic (simplified)
        min_required_mm = 1.0
        if 50 <= panel_area_cm2 < 100:
            min_required_mm = 2.0
        elif 100 <= panel_area_cm2 < 500:
            min_required_mm = 4.0
        elif panel_area_cm2 >= 500:
            min_required_mm = 6.0

        if height_mm < min_required_mm:
            self.log_violation(
                rule="Rule 7",
                description=f"Font height {height_mm:.1f}mm is below minimum {min_required_mm}mm for panel area {panel_area_cm2}cm2.",
                penalty=self.PENALTY_MAJOR
            )
            return False
        return True

    def generate_report(self):
        return {
            "status": "FAIL" if self.violations else "PASS",
            "violation_count": len(self.violations),
            "violations": self.violations,
            "compounding_penalty_inr": self.total_penalty
        }

# Example Usage:
# engine = LegalMetrologyEngine()
# engine.check_rule_6_1_c_net_quantity("Net Wt 500 gms") # Fails (uses 'gms')
# engine.check_rule_6_1_e_mrp("Price: 50.00") # Fails (missing standard format)
# print(engine.generate_report())
