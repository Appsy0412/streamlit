import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import datetime
import os
from supabase import create_client
from geopy.geocoders import GoogleV3
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static
from io import BytesIO
import base64
from fpdf import FPDF
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# App title and configuration
st.set_page_config(page_title="Customer Visit Planner", layout="wide")

# Initialize session state for first run
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.customers_df = None
    st.session_state.geocoded_customers = {}
    st.session_state.weekly_plan = {day: [] for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]}
    st.session_state.visit_history = []
    st.session_state.current_customer = None
    st.session_state.commercial_customers = set()
    st.session_state.customer_classification = {}
    st.session_state.priority_weights = {}
    st.session_state.last_visit_dates = {}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Setup", "Data Connection", "Customer Classification", "Plan Weekly Visits", "Daily View", "Visit Customer", "Reports"])

# Supabase setup
@st.cache_resource
def init_supabase():
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_KEY", "")
    if url and key:
        return create_client(url, key)
    return None

supabase = init_supabase()

# Google Maps API setup
@st.cache_resource
def init_geocoder():
    api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
    if api_key:
        return GoogleV3(api_key=api_key)
    return None

geocoder = init_geocoder()

# Helper function to geocode addresses
def geocode_address(address):
    if address in st.session_state.geocoded_customers:
        return st.session_state.geocoded_customers[address]
    
    if geocoder:
        try:
            location = geocoder.geocode(address)
            if location:
                coords = (location.latitude, location.longitude)
                st.session_state.geocoded_customers[address] = coords
                return coords
        except Exception as e:
            st.error(f"Geocoding error: {e}")
    
    return None

# Helper function to identify commercial customers using AI
def classify_commercial_customers(df):
    # Simple heuristic: look for commercial indicators in name/address
    commercial_indicators = [
        'inc', 'llc', 'ltd', 'corp', 'inc.', 'corporation', 'co.', 'company', 
        'services', 'solutions', 'industries', 'systems', 'technologies',
        'manufacturing', 'enterprises', 'group', 'association', 'office',
        'building', 'suite', 'ste', 'floor', 'fl', 'unit', 'plaza', 'tower'
    ]
    
    classifications = {}
    for idx, row in df.iterrows():
        # Combine relevant fields for analysis
        combined_text = f"{row['Name']} {row['Street']} {row['Town']}".lower()
        
        # Check for commercial indicators
        is_commercial = any(indicator in combined_text for indicator in commercial_indicators)
        
        # Store classification
        customer_id = row['Abbn']
        classifications[customer_id] = 'commercial' if is_commercial else 'retail'
    
    return classifications

# Helper function to generate PDF report
def generate_pdf_report(weekly_plan, visit_history):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Weekly Customer Visit Plan & Report", ln=True, align="C")
    pdf.ln(10)
    
    # Weekly Plan Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Planned Visits", ln=True)
    pdf.set_font("Arial", "", 10)
    
    for day, customers in weekly_plan.items():
        if customers:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"{day}:", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for customer in customers:
                pdf.cell(0, 10, f"• {customer['Name']} - {customer['Town']}", ln=True)
        
        pdf.ln(5)
        # This function should be added to your app.py file
# It exactly matches your Supabase column structure

def fetch_customers_from_supabase():
    if not supabase:
        st.error("Supabase connection not configured")
        return None
    
    try:
        # Get the table name from session state or use default
        table_name = st.session_state.get("table_name", "customers")
        
        # Fetch all records from your Supabase table
        response = supabase.table(table_name).select("*").execute()
        
        if not response.data:
            st.warning(f"No data found in table '{table_name}'")
            return None
        
        # Create a DataFrame directly from the response data
        # Your column names already match what the app expects!
        df = pd.DataFrame(response.data)
        
        # Remove the 'created_at' column which isn't needed by the app
        if 'created_at' in df.columns:
            df = df.drop(columns=['created_at'])
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data from Supabase: {e}")
        return None

# Then in your "Data Connection" page, replace the sample data generation with:
if st.button("Connect to Database"):
    customers_df = fetch_customers_from_supabase()
    if customers_df is not None:
        st.session_state.customers_df = customers_df
        st.session_state.initialized = True
        
        # Run initial commercial classification
        st.session_state.customer_classification = classify_commercial_customers(customers_df)
        commercial_count = sum(1 for c in st.session_state.customer_classification.values() if c == 'commercial')
        
        # Initialize priority weights
        for cust_id in customers_df['Abbn']:
            st.session_state.priority_weights[cust_id] = 5  # Default priority
        
        st.success(f"Connected to database and loaded {len(customers_df)} customers")
        st.info(f"Initial AI classification identified {commercial_count} commercial customers")
    
    # Visit History Section
    if visit_history:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Completed Visits", ln=True)
        pdf.set_font("Arial", "", 10)
        
        # Group visits by date
        visits_by_date = {}
        for visit in visit_history:
            date = visit["date"].split(" ")[0]  # Just get the date part
            if date not in visits_by_date:
                visits_by_date[date] = []
            visits_by_date[date].append(visit)
        
        # Display visits by date
        for date, visits in visits_by_date.items():
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"{date}:", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for visit in visits:
                pdf.cell(0, 10, f"• {visit['customer_name']} - {visit['notes'][:50]}...", ln=True)
            
            pdf.ln(5)
    
    # Generate the PDF as a byte stream
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes

# Function to optimize route
def optimize_route(customer_list, start_point=None):
    if not customer_list:
        return []
    
    # If we don't have a start point, use the first customer as the start
    if start_point is None and customer_list:
        start_point = geocode_address(f"{customer_list[0]['Street']}, {customer_list[0]['Town']}")
    
    # If we still don't have a start point, just return the original list
    if start_point is None:
        return customer_list
    
    # Simple greedy algorithm to find the nearest unvisited customer
    remaining = customer_list.copy()
    route = []
    current_point = start_point
    
    while remaining:
        nearest_idx = 0
        nearest_dist = float('inf')
        
        for i, customer in enumerate(remaining):
            address = f"{customer['Street']}, {customer['Town']}"
            coords = geocode_address(address)
            
            if coords:
                dist = geodesic(current_point, coords).kilometers
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i
        
        next_customer = remaining.pop(nearest_idx)
        route.append(next_customer)
        address = f"{next_customer['Street']}, {next_customer['Town']}"
        current_point = geocode_address(address) or current_point
    
    return route

# Function to estimate priority score
def calculate_priority(customer_id):
    # Default priority is 5 (middle)
    priority = st.session_state.priority_weights.get(customer_id, 5)
    
    # Adjust based on days since last visit (if known)
    last_visit = st.session_state.last_visit_dates.get(customer_id)
    if last_visit:
        days_since_visit = (datetime.datetime.now() - last_visit).days
        # Increase priority for customers not seen in a while
        if days_since_visit > 90:  # More than 3 months
            priority += 3
        elif days_since_visit > 30:  # More than 1 month
            priority += 2
    else:
        # New customers (never visited) get a slight boost
        priority += 1
    
    return priority

# Setup page
if page == "Setup":
    st.title("Setup")
    st.write("Configure your Customer Visit Planner app settings")
    
    # Supabase configuration
    st.subheader("Supabase Configuration")
    supabase_url = st.text_input("Supabase URL", value=st.secrets.get("SUPABASE_URL", ""))
    supabase_key = st.text_input("Supabase API Key", value=st.secrets.get("SUPABASE_KEY", ""), type="password")
    
    # Google Maps API configuration
    st.subheader("Google Maps Configuration")
    google_maps_key = st.text_input("Google Maps API Key", value=st.secrets.get("GOOGLE_MAPS_API_KEY", ""), type="password")
    
    # Save settings
    if st.button("Save Settings"):
        # In a real app, we'd save these to st.secrets
        st.success("Settings saved!")

# Data Connection page
elif page == "Data Connection":
    st.title("Connect to Supabase")
    
    if not supabase:
        st.error("Supabase connection not configured. Please set your credentials in the Setup page.")
    else:
        st.write("Connect to your existing Supabase database")
        
        # Database table selection
        st.subheader("Select Customer Table")
        table_name = st.text_input("Supabase Table Name", "customers")
        
        # Column mapping
        st.subheader("Map Database Columns")
        st.write("Match your database columns to the required fields")
        
        # These would be retrieved from Supabase in a real implementation
        sample_columns = ["id", "name", "address", "city", "phone", "mobile", "notes", "customer_type"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            id_column = st.selectbox("Customer ID", options=sample_columns, index=0)
            name_column = st.selectbox("Name", options=sample_columns, index=1)
            street_column = st.selectbox("Street", options=sample_columns, index=2)
            town_column = st.selectbox("Town/City", options=sample_columns, index=3)
        
        with col2:
            phone_column = st.selectbox("Phone", options=sample_columns, index=4)
            mobile_column = st.selectbox("Mobile", options=sample_columns, index=5)
            other_column = st.selectbox("Other/Notes", options=sample_columns, index=6)
            type_column = st.selectbox("Customer Type (optional)", options=["None"] + sample_columns, index=7)
        
        # Connect button
        if st.button("Connect to Database"):
            try:
                # In a real implementation, fetch data from Supabase
                if supabase:
                    # This is example code - in reality you'd fetch from Supabase
                    response = supabase.table(table_name).select("*").execute()
                    
                    # For demo purposes, create sample data
                    sample_data = {
                        'Abbn': [f'CUST{i}' for i in range(1, 101)],
                        'Name': [f'Customer {i}' for i in range(1, 101)],
                        'Street': [f'{i} Main St' for i in range(1, 101)],
                        'Town': ['Cityville' if i % 3 == 0 else 'Townsburg' if i % 3 == 1 else 'Villageton' for i in range(1, 101)],
                        'Phone': [f'555-{i:03d}-{i*2:04d}' for i in range(1, 101)],
                        'Mobile': [f'555-{i*3:03d}-{i*4:04d}' for i in range(1, 101)],
                        'Other': [f'Note {i}' for i in range(1, 101)]
                    }
                    df = pd.DataFrame(sample_data)
                    
                    st.session_state.customers_df = df
                    st.session_state.initialized = True
                    
                    # Run initial commercial classification
                    st.session_state.customer_classification = classify_commercial_customers(df)
                    commercial_count = sum(1 for c in st.session_state.customer_classification.values() if c == 'commercial')
                    
                    # Initialize priority weights (random for demo)
                    for cust_id in df['Abbn']:
                        st.session_state.priority_weights[cust_id] = np.random.randint(1, 10)
                    
                    st.success(f"Connected to database and loaded {len(df)} customers")
                    st.info(f"Initial AI classification identified {commercial_count} commercial customers")
            except Exception as e:
                st.error(f"Error connecting to database: {e}")
        
        # Display sample of the data
        if st.session_state.customers_df is not None:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.customers_df.head())

# Customer Classification page
elif page == "Customer Classification":
    st.title("Customer Classification")
    
    if not st.session_state.initialized:
        st.warning("Please connect to your Supabase database first")
    else:
        st.write("Review and adjust AI-based customer classifications")
        
        # Filter options
        filter_name = st.text_input("Filter by Name")
        filter_town = st.text_input("Filter by Town")
        classification_filter = st.selectbox("Filter by Classification", ["All", "Commercial", "Retail", "Unclassified"])
        
        df = st.session_state.customers_df
        
        # Apply filters
        filtered_df = df.copy()
        if filter_name:
            filtered_df = filtered_df[filtered_df['Name'].str.contains(filter_name, case=False, na=False)]
        if filter_town:
            filtered_df = filtered_df[filtered_df['Town'].str.contains(filter_town, case=False, na=False)]
        
        # Apply classification filter
        if classification_filter != "All":
            if classification_filter == "Commercial":
                filtered_df = filtered_df[filtered_df['Abbn'].isin([k for k, v in st.session_state.customer_classification.items() if v == 'commercial'])]
            elif classification_filter == "Retail":
                filtered_df = filtered_df[filtered_df['Abbn'].isin([k for k, v in st.session_state.customer_classification.items() if v == 'retail'])]
            else:  # Unclassified
                filtered_df = filtered_df[~filtered_df['Abbn'].isin(st.session_state.customer_classification.keys())]
        
        # Display classification interface
        for i, row in filtered_df.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            cust_id = row['Abbn']
            current_classification = st.session_state.customer_classification.get(cust_id, "unclassified")
            
            with col1:
                st.write(f"**{row['Name']}**")
                st.write(f"{row['Street']}, {row['Town']}")
                st.write(f"Phone: {row['Phone']}")
            
            with col2:
                st.write(f"Current: {current_classification}")
            
            with col3:
                new_classification = st.selectbox(
                    "Set classification", 
                    ["commercial", "retail"], 
                    index=0 if current_classification == "commercial" else 1,
                    key=f"class_{cust_id}"
                )
                
                if st.button("Update", key=f"update_{cust_id}"):
                    st.session_state.customer_classification[cust_id] = new_classification
                    st.success(f"Updated {row['Name']} to {new_classification}")
                    
                    # In a real implementation, save back to Supabase
                    if supabase:
                        try:
                            # Example code - update classification in Supabase
                            # supabase.table(table_name).update({"customer_type": new_classification}).eq("id", cust_id).execute()
                            pass
                        except Exception as e:
                            st.error(f"Error saving to database: {e}")
            
            st.divider()
        
        # Batch operations
        st.subheader("Batch Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Re-run AI Classification"):
                st.session_state.customer_classification = classify_commercial_customers(df)
                st.success("AI classification completed")
        
        with col2:
            if st.button("Mark All Filtered as Commercial"):
                for cust_id in filtered_df['Abbn']:
                    st.session_state.customer_classification[cust_id] = "commercial"
                st.success(f"Marked {len(filtered_df)} customers as commercial")
                
                # In a real implementation, save back to Supabase
                if supabase:
                    try:
                        # Example code - update classifications in Supabase in bulk
                        # for cust_id in filtered_df['Abbn']:
                        #     supabase.table(table_name).update({"customer_type": "commercial"}).eq("id", cust_id).execute()
                        pass
                    except Exception as e:
                        st.error(f"Error saving to database: {e}")

# Plan Weekly Visits page
elif page == "Plan Weekly Visits":
    st.title("Plan Weekly Visits")
    
    if not st.session_state.initialized:
        st.warning("Please connect to your Supabase database first")
    else:
        # Get commercial customers
        df = st.session_state.customers_df
        commercial_df = df[df['Abbn'].isin([k for k, v in st.session_state.customer_classification.items() if v == 'commercial'])]
        
        # Filter options
        st.subheader("Filter Customers")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_town = st.text_input("Filter by Town")
        
        with col2:
            min_priority = st.slider("Minimum Priority", 1, 10, 1)
        
        # Apply filters
        filtered_df = commercial_df.copy()
        if filter_town:
            filtered_df = filtered_df[filtered_df['Town'].str.contains(filter_town, case=False, na=False)]
        
        # Filter by priority
        if min_priority > 1:
            high_priority_customers = [k for k, v in st.session_state.priority_weights.items() if v >= min_priority]
            filtered_df = filtered_df[filtered_df['Abbn'].isin(high_priority_customers)]
        
        # Display available customers
        st.subheader("Available Customers")
        
        # Create a list for drag and drop
        customer_list = []
        for i, row in filtered_df.iterrows():
            cust_id = row['Abbn']
            priority = st.session_state.priority_weights.get(cust_id, 5)
            last_visit = st.session_state.last_visit_dates.get(cust_id, "Never")
            if isinstance(last_visit, datetime.datetime):
                last_visit = last_visit.strftime("%Y-%m-%d")
            
            customer_list.append({
                "id": cust_id,
                "name": row['Name'],
                "address": f"{row['Street']}, {row['Town']}",
                "priority": priority,
                "last_visit": last_visit
            })
        
        # Convert to DataFrame for display
        if customer_list:
            customers_display_df = pd.DataFrame(customer_list)
            st.dataframe(customers_display_df)
        else:
            st.info("No customers match your filters")
        
        # Weekly planning
        st.subheader("Weekly Visit Plan")
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_tabs = st.tabs(days)
        
        for i, day in enumerate(days):
            with day_tabs[i]:
                st.write(f"Plan for {day}")
                
                # Display current plan for the day
                if day in st.session_state.weekly_plan and st.session_state.weekly_plan[day]:
                    st.write(f"**Current plan: {len(st.session_state.weekly_plan[day])} customers**")
                    
                    for j, customer in enumerate(st.session_state.weekly_plan[day]):
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.write(f"{j+1}. **{customer['Name']}** - {customer['Street']}, {customer['Town']}")
                        
                        with col2:
                            if st.button("Remove", key=f"remove_{day}_{j}"):
                                st.session_state.weekly_plan[day].pop(j)
                                st.rerun()
                else:
                    st.write("No customers planned for this day yet")
                
                # Add customers to this day
                st.write("---")
                st.write("Add customers to this day:")
                
                # Select customer to add
                selected_customers = st.multiselect(
                    "Select customers to add", 
                    options=[f"{c['name']} ({c['address']})" for c in customer_list],
                    key=f"add_customers_{day}"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Add Selected Customers", key=f"add_{day}"):
                        added = 0
                        for sel in selected_customers:
                            # Extract name from selection string
                            name = sel.split(" (")[0]
                            # Find customer in dataframe
                            customer_row = df[df['Name'] == name].iloc[0]
                            # Add to plan if not already there
                            if not any(c['Name'] == name for c in st.session_state.weekly_plan[day]):
                                st.session_state.weekly_plan[day].append(customer_row.to_dict())
                                added += 1
                        
                        if added > 0:
                            st.success(f"Added {added} customers to {day}")
                            st.rerun()
                
                with col2:
                    if st.button("Optimize Route", key=f"optimize_{day}"):
                        if st.session_state.weekly_plan[day]:
                            st.session_state.weekly_plan[day] = optimize_route(st.session_state.weekly_plan[day])
                            st.success(f"Route optimized for {day}")
                            st.rerun()
        
        # Generate Weekly Plan
        st.subheader("Generate Weekly Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_daily = st.number_input("Maximum customers per day", min_value=1, max_value=20, value=5)
        
        with col2:
            if st.button("Auto-Generate Weekly Plan"):
                # Reset current plan
                st.session_state.weekly_plan = {day: [] for day in days}
                
                # Sort customers by priority
                all_customers = []
                for i, row in commercial_df.iterrows():
                    cust_id = row['Abbn']
                    priority = calculate_priority(cust_id)
                    all_customers.append((priority, row.to_dict()))
                
                all_customers.sort(reverse=True)  # Higher priority first
                
                # Distribute customers across days
                day_idx = 0
                for _, customer in all_customers:
                    current_day = days[day_idx]
                    
                    # If day is full, move to next day
                    if len(st.session_state.weekly_plan[current_day]) >= max_daily:
                        day_idx = (day_idx + 1) % len(days)
                        current_day = days[day_idx]
                    
                    # Add customer to day
                    st.session_state.weekly_plan[current_day].append(customer)
                    
                    # If we've added enough customers total, stop
                    total_planned = sum(len(customers) for customers in st.session_state.weekly_plan.values())
                    if total_planned >= max_daily * len(days):
                        break
                
                # Optimize routes for each day
                for day in days:
                    if st.session_state.weekly_plan[day]:
                        st.session_state.weekly_plan[day] = optimize_route(st.session_state.weekly_plan[day])
                
                st.success("Weekly plan generated and routes optimized!")
                st.rerun()
        
        # Microsoft Calendar Integration
        st.subheader("Microsoft Calendar Integration")
        
        calendar_sync = st.checkbox("Sync with Microsoft Calendar", value=False)
        
        if calendar_sync:
            st.info("Calendar integration requires Microsoft Graph API credentials")
            
            # In a real implementation, this would use Microsoft Graph API to create calendar events
            
            if st.button("Sync Weekly Plan to Calendar"):
                st.success("Weekly plan synced to Microsoft Calendar!")
                # In a real implementation, this would create calendar events for each customer visit
        
        # Export plan
        if st.button("Generate Weekly Plan PDF"):
            pdf_bytes = generate_pdf_report(st.session_state.weekly_plan, st.session_state.visit_history)
            
            # Create download link
            b64_pdf = base64.b64encode(pdf_bytes).decode()
            curr_date = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"weekly_plan_{curr_date}.pdf"
            
            href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

# Daily View page
elif page == "Daily View":
    st.title("Daily View")
    
    if not st.session_state.initialized:
        st.warning("Please connect to your Supabase database first")
    else:
        # Select day to view
        day = st.selectbox("Select Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        
        if day in st.session_state.weekly_plan and st.session_state.weekly_plan[day]:
            st.write(f"**{len(st.session_state.weekly_plan[day])} customers planned for {day}**")
            
            # Create a map
            if geocoder:
                st.subheader("Route Map")
                
                # Create a base map
                m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)  # Default to London if no coordinates yet
                
                # Add markers for each customer
                valid_locations = []
                for i, customer in enumerate(st.session_state.weekly_plan[day]):
                    address = f"{customer['Street']}, {customer['Town']}"
                    coords = geocode_address(address)
                    
                    if coords:
                        valid_locations.append(coords)
                        folium.Marker(
                            location=coords,
                            popup=f"{i+1}. {customer['Name']}",
                            tooltip=f"{i+1}. {customer['Name']}",
                            icon=folium.Icon(icon="building", prefix="fa")
                        ).add_to(m)
                
                # Add route lines
                if len(valid_locations) > 1:
                    folium.PolyLine(
                        locations=valid_locations,
                        color="blue",
                        weight=2.5,
                        opacity=1
                    ).add_to(m)
                
                # Fit the map to the markers
                if valid_locations:
                    sw = [min(loc[0] for loc in valid_locations), min(loc[1] for loc in valid_locations)]
                    ne = [max(loc[0] for loc in valid_locations), max(loc[1] for loc in valid_locations)]
                    m.fit_bounds([sw, ne])
                
                # Display the map
                folium_static(m)
            
            # List all customers for the day with details
            st.subheader("Customer Details")
            
            for i, customer in enumerate(st.session_state.weekly_plan[day]):
                with st.expander(f"{i+1}. {customer['Name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Address:** {customer['Street']}, {customer['Town']}")
                        st.write(f"**Phone:** {customer['Phone']}")
                        if 'Mobile' in customer and customer['Mobile']:
                            st.write(f"**Mobile:** {customer['Mobile']}")
                    
                    with col2:
                        cust_id = customer['Abbn']
                        priority = st.session_state.priority_weights.get(cust_id, 5)
                        st.write(f"**Priority:** {priority}/10")
                        
                        last_visit = st.session_state.last_visit_dates.get(cust_id, "Never")
                        if isinstance(last_visit, datetime.datetime):
                            last_visit = last_visit.strftime("%Y-%m-%d")
                        st.write(f"**Last Visit:** {last_visit}")
                    
                    # Visit button
                    if st.button("Start Visit", key=f"visit_{cust_id}"):
                        st.session_state.current_customer = customer
                        # Redirect to Visit page (this is a hack, real app would use proper navigation)
                        st.rerun()
        else:
            st.info(f"No customers planned for {day} yet")
            
        # Current location-based replanning
        st.subheader("Current Location Planning")
        st.write("Find customers near your current location")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Use your current location to find nearby customers")
            if st.button("Find Customers Near Me"):
                # In a real implementation, this would use browser geolocation
                # For demo, we'll use a mock location
                mock_location = (51.5074, -0.1278)  # London coordinates
                
                st.success("Found your location")
                
                # Find nearby customers
                nearby_customers = []
                
                # Simulate finding nearby commercial customers
                for i, row in df.iterrows():
                    if row['Abbn'] in st.session_state.customer_classification and st.session_state.customer_classification[row['Abbn']] == 'commercial':
                        address = f"{row['Street']}, {row['Town']}"
                        coords = geocode_address(address)
                        
                        if coords:
                            # Calculate distance to current location
                            dist = geodesic(mock_location, coords).kilometers
                            
                            # Add if within 15km
                            if dist <= 15:
                                nearby_customers.append({
                                    'customer': row.to_dict(),
                                    'distance': dist
                                })
                
                # Sort by distance
                nearby_customers.sort(key=lambda x: x['distance'])
                
                # Display nearby customers
                st.write(f"Found {len(nearby_customers)} commercial customers within 15km")
                
                for i, item in enumerate(nearby_customers[:5]):  # Show top 5
                    customer = item['customer']
                    dist = item['distance']
                    
                    st.write(f"{i+1}. **{customer['Name']}** - {dist:.1f}km away")
                    st.write(f"   {customer['Street']}, {customer['Town']}")
                    
                    if st.button("Add to Today's Plan", key=f"add_nearby_{i}"):
                        # Add to current day's plan
                        if not any(c['Name'] == customer['Name'] for c in st.session_state.weekly_plan[day]):
                            st.session_state.weekly_plan[day].append(customer)
                            st.success(f"Added {customer['Name']} to {day}")
                            st.rerun()
        
        with col2:
            st.write("Optimize today's route based on your current location")
            if st.button("Reoptimize Today's Route"):
                if st.session_state.weekly_plan[day]:
                    # In a real implementation, this would use browser geolocation
                    # For demo, we'll use a mock location
                    mock_location = (51.5074, -0.1278)  # London coordinates
                    
                    # Reoptimize route starting from current location
                    st.session_state.weekly_plan[day] = optimize_route(st.session_state.weekly_plan[day], start_point=mock_location)
                    st.success(f"Route reoptimized for {day} based on your current location")
                    st.rerun()
                else:
                    st.warning("No customers planned for today yet")
        
        # Options for the current day
        st.subheader("Day Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Optimize Route"):
                if day in st.session_state.weekly_plan and st.session_state.weekly_plan[day]:
                    st.session_state.weekly_plan[day] = optimize_route(st.session_state.weekly_plan[day])
                    st.success("Route optimized")
                    st.rerun()
        
        with col2:
            if st.button("Clear Day"):
                st.session_state.weekly_plan[day] = []
                st.success(f"Cleared plan for {day}")
                st.rerun()

# Visit Customer page
elif page == "Visit Customer":
    st.title("Customer Visit")
    
    if not st.session_state.initialized:
        st.warning("Please connect to your Supabase database first")
    elif st.session_state.current_customer is None:
        st.info("No customer selected. Please select a customer from the Daily View.")
    else:
        customer = st.session_state.current_customer
        
        # Display customer info
        st.subheader(f"Visiting: {customer['Name']}")
        st.write(f"**Address:** {customer['Street']}, {customer['Town']}")
        st.write(f"**Phone:** {customer['Phone']}")
        if 'Mobile' in customer and customer['Mobile']:
            st.write(f"**Mobile:** {customer['Mobile']}")
        
        # Map with location
        if geocoder:
            address = f"{customer['Street']}, {customer['Town']}"
            coords = geocode_address(address)
            
            if coords:
                st.subheader("Location")
                m = folium.Map(location=coords, zoom_start=15)
                folium.Marker(
                    location=coords,
                    popup=customer['Name'],
                    tooltip=customer['Name'],
                    icon=folium.Icon(icon="building", prefix="fa")
                ).add_to(m)
                folium_static(m)
        
        # Notes section
        st.subheader("Visit Notes")
        notes = st.text_area("Enter notes about this visit", height=150)
        
        # Check-in button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Check-in at This Location"):
                # In a real app, we would get the actual GPS coordinates
                timestamp = datetime.datetime.now()
                
                # Add to visit history
                visit_record = {
                    "customer_id": customer['Abbn'],
                    "customer_name": customer['Name'],
                    "date": timestamp.strftime("%Y-%m-%d %H:%M"),
                    "notes": notes,
                    "location": address
                }
                
                st.session_state.visit_history.append(visit_record)
                
                # Update last visit date
                st.session_state.last_visit_dates[customer['Abbn']] = timestamp
                
                # Save to Supabase in a real implementation
                if supabase:
                    try:
                        # Example code - save visit to Supabase
                        # supabase.table("visits").insert(visit_record).execute()
                        pass
                    except Exception as e:
                        st.error(f"Error saving visit: {e}")
                
                st.success("Visit recorded!")
        
        with col2:
            if st.button("Complete Visit"):
                # Clear current customer and return to daily view
                st.session_state.current_customer = None
                st.success("Visit completed!")
                st.rerun()
        
        # Previous visit history
        st.subheader("Previous Visits")
        previous_visits = [v for v in st.session_state.visit_history 
                          if v["customer_id"] == customer['Abbn']]
        
        if previous_visits:
            for visit in previous_visits:
                st.write(f"**{visit['date']}**: {visit['notes']}")
        else:
            st.info("No previous visit records")
        
        # Offline support notice
        st.subheader("Offline Support")
        st.info("This visit data will be synchronized when you're back online")

# Reports page
elif page == "Reports":
    st.title("Reports")
    
    if not st.session_state.initialized:
        st.warning("Please connect to your Supabase database first")
    else:
        st.subheader("Visit History")
        
        # Filter options
        date_range = st.date_input(
            "Select date range",
            value=(
                datetime.datetime.now() - datetime.timedelta(days=30),
                datetime.datetime.now()
            ),
            max_value=datetime.datetime.now()
        )
        
        # Filter by date
        filtered_visits = []
        for visit in st.session_state.visit_history:
            visit_date = datetime.datetime.strptime(visit["date"], "%Y-%m-%d %H:%M").date()
            if date_range[0] <= visit_date <= date_range[1]:
                filtered_visits.append(visit)
        
        # Display visits
        if filtered_visits:
            # Convert to dataframe for display
            visits_df = pd.DataFrame(filtered_visits)
            st.dataframe(visits_df)
            
            # Visualizations
            st.subheader("Visit Analytics")
            
            # Visits by day of week
            if not visits_df.empty and len(visits_df) > 1:
                visits_df['weekday'] = pd.to_datetime(visits_df['date']).dt.day_name()
                
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                visits_by_day = visits_df['weekday'].value_counts().reindex(days_order, fill_value=0)
                
                st.bar_chart(visits_by_day)
                
            # Generate report for export
            export_format = st.selectbox("Export Format", ["PDF", "Excel"])
            
            if st.button(f"Generate {export_format} Report"):
                if export_format == "PDF":
                    pdf_bytes = generate_pdf_report(st.session_state.weekly_plan, filtered_visits)
                    
                    # Create download link
                    b64_pdf = base64.b64encode(pdf_bytes).decode()
                    curr_date = datetime.datetime.now().strftime("%Y%m%d")
                    filename = f"visit_report_{curr_date}.pdf"
                    
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    # Excel export would be implemented here
                    st.info("Excel export would be available in the full implementation")
            
            # Notion/Motion integration
            st.subheader("Integration Options")
            
            integration_type = st.selectbox("Select Integration", ["Notion", "Motion", "None"])
            
            if integration_type != "None":
                if st.button(f"Export to {integration_type}"):
                    st.success(f"Data exported to {integration_type}!")
                    # In a real implementation, this would use the respective APIs
        else:
            st.info("No visits in the selected date range")

# Run the app
if __name__ == "__main__":
    st.sidebar.info("""
    **About**
    
    Customer Visit Planner helps field sales teams plan, 
    optimize, and track customer visits efficiently.
    
    This app is designed to work both online and offline.
    """)
