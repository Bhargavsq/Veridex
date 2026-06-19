import streamlit as st

from shopping_agent import (
    run_research_agent,
    compare_products,
    get_search_history,
    delete_topic,
    AVAILABLE_MODELS
)

# PAGE
st.set_page_config(
    page_title="",
    page_icon="🤖",
    layout="wide"
)
st.markdown("""
    <style>
    /* Eliminate the huge default padding at the top of the main page area */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
    }
    /* Force long text to cut off cleanly with an ellipsis (...) instead of breaking lines */
    .sidebar-btn-text {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# HEADER
st.markdown("<h1 style='margin-top: 0px; margin-bottom: 0px;'>Veridex</h1>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title(
    "Search History"
)
history = get_search_history()
st.sidebar.caption(
    f"{len(history)} searches"
)
for item in history[:30]:
    col1, col2 = st.sidebar.columns([4,1])
    with col1:
        button_text = item[:40] + "..." if len(item) > 40 else item

        if st.button(
            button_text,
            key=f"open_{item}",
            use_container_width=True,
            help=item  # <-- THIS SHOWS THE FULL TEXT ON HOVER
        ):
            st.session_state["product"] = item
            st.session_state["result"] = (
                run_research_agent(item)
            )
            st.rerun()
    with col2:
        if st.button(
            "❌",
            key=f"delete_{item}"
        ):
            delete_topic(item)
            st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button(
    "New Search"
):
    st.session_state.pop(
        "result",
        None
    )
    st.session_state.pop(
        "product",
        None
    )
    st.rerun()

# SEARCH AREA
default_product = (
    st.session_state.get("product","")
)

col1, col2, col3 = st.columns([4, 2, 1], vertical_alignment="bottom")
with col1:
    product = st.text_input(
        "Search Label Hidden",
        value=default_product,
        placeholder="Ask Agent",
        label_visibility="collapsed",
    )

with col2:  
    selected_model = st.selectbox(
        "Model Label Hidden",
        list(AVAILABLE_MODELS.keys()),
        label_visibility="collapsed"
    )  

with col3:
    search_btn = st.button(
        "Search",
        use_container_width=True,
        type="primary"
    )

# OPTIONS
ignore_memory = st.checkbox(
    "Ignore Memory (Force Fresh Search)"
)

# SEARCH
if search_btn and product:
    with st.spinner("Researching..."):
        if " vs " in product.lower():
            product1, product2 = product.split(
                " vs ",
                1
            )
            result = compare_products(
                product1.strip(),
                product2.strip(),
                selected_model
            )
        else:
            result = run_research_agent(
                product,
                use_memory=not ignore_memory,
                model_choice=selected_model
            )

        st.session_state["result"] = result

# RESULTS
if "result" in st.session_state:
    st.subheader("Analysis")
    with st.container(height=400, border=True):
        st.markdown(st.session_state["result"])