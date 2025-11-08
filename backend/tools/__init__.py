TOOLS_INIT = """
\"\"\"Agent tools package\"\"\"
from .rfp_tools import find_rfp_online
from .sku_tools import match_sku, estimate_cost

__all__ = ['find_rfp_online', 'match_sku', 'estimate_cost']
"""