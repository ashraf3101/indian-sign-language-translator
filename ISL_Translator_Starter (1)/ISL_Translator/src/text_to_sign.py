from pathlib import Path
import streamlit as st

SUPPORTED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")

def render_spelling(text: str, signs_dir: Path):
    normalized = text.upper()
    st.write("Sign sequence (letter-by-letter):")
    for start in range(0, len(normalized), 8):
        chunk = normalized[start:start + 8]
        cols = st.columns(min(len(chunk), 8))
        for col, char in zip(cols, chunk):
            with col:
                if char == " ":
                    st.markdown("**SPACE**")
                    st.caption("Word break")
                elif char not in SUPPORTED:
                    st.write(char)
                    st.caption("Unsupported")
                else:
                    image_path = signs_dir / f"{char}.png"
                    if image_path.exists():
                        st.image(str(image_path), caption=char, use_container_width=True)
                    else:
                        st.markdown(f"### {char}")
                        st.caption("Add assets/signs/" + f"{char}.png")
    st.caption(
        "Use verified ISL alphabet sign images. This spells letters; it does not produce "
        "natural ISL grammar or word-level signs."
    )
