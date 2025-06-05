import streamlit as st
import pandas as pd
import numpy as np

# Title of the application
st.title("My First Streamlit App")

# Display some text
st.write("Hello, Streamlit! This is a simple interactive application.")

# Add a text input widget
user_name = st.text_input("What's your name?", "Guest")
st.write(f"Hello, {user_name}!")

# Add a slider widget
x = st.slider("Select a value for x", 0, 100, 50)
st.write(f"You selected: {x}")

# Add a checkbox
if st.checkbox("Show a DataFrame"):
    st.subheader("Random Data")
    # Create a random DataFrame
    data = pd.DataFrame(
        np.random.randn(10, 5),
        columns=('col %d' % i for i in range(5))
    )
    st.dataframe(data) # Display the DataFrame

# Add a button
if st.button("Say Goodbye"):
    st.write("Goodbye! Thanks for using the app.")

# Display a simple plot
st.subheader("Simple Line Chart")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
).cumsum()
st.line_chart(chart_data)

# Show code (useful for demonstrating your app's code)
st.code("""
import streamlit as st
import pandas as pd
import numpy as np

st.title("My First Streamlit App")
# ... (rest of your code)
""")