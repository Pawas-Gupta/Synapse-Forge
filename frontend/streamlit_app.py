"""Streamlit Frontend for RFP Agent System"""
import streamlit as st
import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Backend API configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:5000/api')

# Initialize Groq API key for local agent execution if needed
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY', '')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def call_api(endpoint: str, method: str = 'GET', data: dict = None):
    """Call backend API"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=30)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=120)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {'status': 'error', 'error': str(e)}


def format_currency(amount: float) -> str:
    """Format currency values"""
    return f"${amount:,.2f}"


def display_product_card(product: dict):
    """Display product information card"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{product['name']}**")
            st.caption(product['description'])
            st.caption(f"SKU: {product['sku_id']} | Category: {product.get('category', 'N/A')}")
        
        with col2:
            st.metric("Unit Cost", format_currency(product['unit_cost']))
            if 'match_score' in product:
                st.metric("Match", f"{product['match_score']}%")


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI RFP Response Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://via.placeholder.com/200x80/4A90E2/FFFFFF?text=RFP+Agent", use_column_width=True)
    
    st.header("🔧 System Configuration")
    
    st.markdown("### Tech Stack")
    st.code("""
🧠 LLM: Groq (GPT-OSS-20B)
🔗 Framework: LangChain
⚙️  Backend: Flask API
🎨 Frontend: Streamlit
💾 Database: SQLite
📊 Analytics: Pandas
    """)
    
    st.markdown("### Agent Architecture")
    st.markdown("""
    - 🎯 **Main Agent**: Orchestrator
    - 📋 **Sales Agent**: RFP Discovery
    - 🔧 **Technical Agent**: SKU Matching
    - 💰 **Pricing Agent**: Cost Estimation
    """)
    
    st.divider()
    
    # API Status
    st.markdown("### 🔌 Backend Status")
    health_check = call_api('/health')
    
    if health_check.get('status') == 'healthy':
        st.success("✅ Backend Connected")
        st.caption(f"Version: {health_check.get('version', 'N/A')}")
    else:
        st.error("❌ Backend Disconnected")
        st.warning("Start backend: `python backend/app.py`")
    
    st.divider()
    
    # Quick Actions
    if st.button("🔄 Refresh Catalog"):
        st.rerun()
    
    if st.button("📖 View Documentation"):
        st.info("Documentation: See README.md")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🤖 AI RFP Response Generator")
st.markdown("**Automated Multi-Agent System for B2B Manufacturing RFP Responses**")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 RFP Analysis",
    "📦 Product Catalog",
    "🔍 Individual Agents",
    "📊 Results History"
])


# ============================================================================
# TAB 1: RFP ANALYSIS
# ============================================================================

with tab1:
    st.header("Complete RFP Analysis Workflow")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 RFP Input")
        rfp_input = st.text_area(
            "Paste RFP document or enter requirements:",
            placeholder="""Example:
Project: Manufacturing facility electrical upgrade
Requirements:
- 10x 100A circuit breakers, 480V
- 3x 50HP motor starters
- 1x 75kVA transformer
- 200ft cable tray system
Timeline: 6-8 weeks
Budget: $60,000-$80,000""",
            height=250
        )
        
        analyze_col1, analyze_col2, analyze_col3 = st.columns([2, 1, 1])
        
        with analyze_col1:
            analyze_button = st.button("🚀 Start Full Analysis", type="primary", use_container_width=True)
        
        with analyze_col2:
            if st.button("💾 Save RFP", use_container_width=True):
                st.info("Save functionality - Coming soon")
        
        with analyze_col3:
            if st.button("📋 Load Sample", use_container_width=True):
                st.session_state.sample_loaded = True
                st.rerun()
    
    with col2:
        st.subheader("📊 Quick Stats")
        catalog_data = call_api('/catalog')
        
        if catalog_data.get('status') == 'success':
            summary = catalog_data.get('summary', {})
            st.metric("Total Products", summary.get('total_skus', 0))
            st.metric("Avg Price", format_currency(summary.get('price_stats', {}).get('average', 0)))
            st.metric("Categories", len(summary.get('categories', {})))
        
        st.info("💡 **Tip**: Paste detailed RFP text for best matching results")
    
    # Analysis Execution
    if analyze_button and rfp_input.strip():
        st.divider()
        st.header("🔄 Agent Workflow Execution")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Execute analysis
        with st.spinner("Initializing agents..."):
            status_text.text("📡 Connecting to backend...")
            progress_bar.progress(10)
            
            result = call_api('/rfp/analyze', method='POST', data={'rfp_input': rfp_input})
            
            if result.get('status') == 'success':
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                st.success("🎉 RFP Analysis Completed Successfully!")
                
                # Display Results
                st.divider()
                st.header("📈 Analysis Results")
                
                # Executive Summary
                final_response = result.get('final_response', {})
                exec_summary = final_response.get('executive_summary', {})
                
                st.subheader("📋 Executive Summary")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Products Matched",
                        len(final_response.get('technical_proposal', {}).get('matched_products', []))
                    )
                
                with col_b:
                    st.metric(
                        "Match Confidence",
                        f"{final_response.get('technical_proposal', {}).get('match_confidence', 0):.1f}%"
                    )
                
                with col_c:
                    st.metric(
                        "Total Investment",
                        format_currency(exec_summary.get('total_investment', 0))
                    )
                
                # Technical Proposal
                st.subheader("🔧 Technical Proposal")
                tech_proposal = final_response.get('technical_proposal', {})
                matched_products = tech_proposal.get('matched_products', [])
                
                if matched_products:
                    for product in matched_products[:5]:  # Show top 5
                        display_product_card(product)
                    
                    if len(matched_products) > 5:
                        with st.expander(f"View all {len(matched_products)} matched products"):
                            for product in matched_products[5:]:
                                display_product_card(product)
                
                # Pricing Breakdown
                st.subheader("💰 Pricing Breakdown")
                pricing = final_response.get('pricing_proposal', {})
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Base Cost", format_currency(pricing.get('total_cost', 0)))
                with col_p2:
                    st.metric("Margin (25%)", format_currency(pricing.get('margin', 0)))
                with col_p3:
                    st.metric("Final Price", format_currency(pricing.get('final_price', 0)), delta=f"+{format_currency(pricing.get('margin', 0))}")
                
                # Breakdown Table
                breakdown = pricing.get('breakdown', [])
                if breakdown:
                    df_breakdown = pd.DataFrame(breakdown)
                    st.dataframe(
                        df_breakdown[['name', 'quantity', 'unit_cost', 'line_total']],
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Complete Workflow Details
                with st.expander("🔍 View Complete Workflow Details"):
                    st.json(result)
                
                # Export Options
                st.divider()
                col_export1, col_export2, col_export3 = st.columns(3)
                
                with col_export1:
                    if st.button("📥 Export to PDF", use_container_width=True):
                        st.info("PDF export - Coming soon")
                
                with col_export2:
                    if st.button("📊 Export to Excel", use_container_width=True):
                        st.info("Excel export - Coming soon")
                
                with col_export3:
                    st.download_button(
                        "💾 Download JSON",
                        data=json.dumps(result, indent=2),
                        file_name="rfp_analysis.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            else:
                progress_bar.progress(0)
                status_text.text("❌ Analysis failed")
                st.error(f"Error: {result.get('error', 'Unknown error')}")


# ============================================================================
# TAB 2: PRODUCT CATALOG
# ============================================================================

with tab2:
    st.header("📦 Product Catalog")
    
    catalog_result = call_api('/catalog')
    
    if catalog_result.get('status') == 'success':
        catalog = catalog_result.get('catalog', [])
        summary = catalog_result.get('summary', {})
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", summary.get('total_skus', 0))
        
        with col2:
            price_stats = summary.get('price_stats', {})
            st.metric("Avg Price", format_currency(price_stats.get('average', 0)))
        
        with col3:
            st.metric("Min Price", format_currency(price_stats.get('minimum', 0)))
        
        with col4:
            st.metric("Max Price", format_currency(price_stats.get('maximum', 0)))
        
        st.divider()
        
        # Category filter
        categories = list(summary.get('categories', {}).keys())
        selected_category = st.selectbox("Filter by Category:", ["All"] + categories)
        
        # Display catalog
        if selected_category == "All":
            display_catalog = catalog
        else:
            display_catalog = [p for p in catalog if p.get('category') == selected_category]
        
        st.subheader(f"Products ({len(display_catalog)})")
        
        for product in display_catalog:
            with st.expander(f"{product['name']} - {format_currency(product['unit_cost'])}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Description:**")
                    st.write(product['description'])
                    st.markdown(f"**SKU:** {product['sku_id']}")
                    st.markdown(f"**Category:** {product.get('category', 'N/A')}")
                
                with col_b:
                    st.markdown("**Specifications:**")
                    st.json(product['specs'])


# ============================================================================
# TAB 3: INDIVIDUAL AGENTS
# ============================================================================

with tab3:
    st.header("🔍 Test Individual Agents")
    st.info("Run agents independently for testing and debugging")
    
    agent_type = st.selectbox(
        "Select Agent:",
        ["Sales Agent", "Technical Agent", "Pricing Agent"]
    )
    
    if agent_type == "Sales Agent":
        st.subheader("📋 Sales Agent - RFP Discovery")
        sales_input = st.text_area("Enter RFP query:", height=150)
        
        if st.button("Run Sales Agent"):
            if sales_input:
                with st.spinner("Running Sales Agent..."):
                    result = call_api('/agents/sales', method='POST', data={'rfp_input': sales_input})
                    st.json(result)
    
    elif agent_type == "Technical Agent":
        st.subheader("🔧 Technical Agent - SKU Matching")
        tech_input = st.text_area("Enter technical requirements:", height=150)
        
        if st.button("Run Technical Agent"):
            if tech_input:
                with st.spinner("Running Technical Agent..."):
                    result = call_api('/agents/technical', method='POST', data={'rfp_text': tech_input})
                    st.json(result)
    
    elif agent_type == "Pricing Agent":
        st.subheader("💰 Pricing Agent - Cost Estimation")
        st.info("First match products using Technical Agent, then run pricing")


# ============================================================================
# TAB 4: RESULTS HISTORY
# ============================================================================

with tab4:
    st.header("📊 Analysis History")
    st.info("History tracking - Coming in next phase")
    
    st.markdown("""
    Future features:
    - Save analysis results
    - Compare multiple RFPs
    - Export reports
    - Analytics dashboard
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("🤖 AI-Powered RFP Response Agent System | Built with Streamlit, Flask, LangChain & Groq")


print("=" * 80)
print("ALL FILES GENERATED SUCCESSFULLY!")
print("=" * 80)