import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from CMGchain import Chain
from CMGportfolio import Portfolio
from CMGutils import clean_text

import os


# Dynamically find correct path
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "Resource", "my_portfolio.csv")

def create_streamlit_app(llm, clean_text):
    st.title("📧 Cold Mail Generator")
    
    # Initialize portfolio inside the function to avoid module-level initialization
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = Portfolio(csv_path)
    
    portfolio = st.session_state.portfolio
    
    # URL input section
    url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-33460")
    submit_button = st.button("Extract Jobs")

    # Initialize session state for jobs
    if 'jobs_data' not in st.session_state:
        st.session_state.jobs_data = []
    if 'url_processed' not in st.session_state:
        st.session_state.url_processed = ""

    if submit_button:
        try:
            with st.spinner("Extracting job postings..."):
                loader = WebBaseLoader([url_input])
                data = clean_text(loader.load().pop().page_content)
                portfolio.load_portfolio()
                jobs = llm.extract_jobs(data)
                
                # Store jobs in session state
                st.session_state.jobs_data = jobs
                st.session_state.url_processed = url_input
                
                if jobs:
                    st.success(f"✅ Found {len(jobs)} job posting(s)!")
                else:
                    st.warning("No job postings found. Please check the URL.")
                    
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

    # Display job selection and email generation
    if st.session_state.jobs_data:
        st.markdown("---")
        st.subheader("📋 Available Job Postings")
        
        # Create tabs if multiple jobs, otherwise show single job
        if len(st.session_state.jobs_data) > 1:
            # Create dropdown for job selection
            job_options = []
            for i, job in enumerate(st.session_state.jobs_data):
                role = job.get('role', 'Unknown Role')
                company = job.get('company_name', 'Unknown Company')
                experience = job.get('experience', 'Not specified')
                job_options.append(f"Job {i+1}: {role} at {company} ({experience})")
            
            selected_job_index = st.selectbox(
                "Select a job posting to generate cold email:",
                range(len(job_options)),
                format_func=lambda x: job_options[x]
            )
            
            selected_job = st.session_state.jobs_data[selected_job_index]
            
        else:
            selected_job = st.session_state.jobs_data[0]
            selected_job_index = 0
        
        # Display selected job details
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📄 Job Details")
            with st.expander("View Full Job Description", expanded=True):
                st.write(f"**Role:** {selected_job.get('role', 'Not specified')}")
                st.write(f"**Company:** {selected_job.get('company_name', 'Not specified')}")
                st.write(f"**Experience:** {selected_job.get('experience', 'Not specified')}")
                st.write(f"**Skills Required:** {selected_job.get('skills', 'Not specified')}")
                
                if selected_job.get('requirements', 'Not specified') != 'Not specified':
                    st.write(f"**Requirements:** {selected_job.get('requirements')}")
                
                if selected_job.get('description', 'Not specified') != 'Not specified':
                    st.write(f"**Description:** {selected_job.get('description')}")
                
                if selected_job.get('preferred_skills', 'Not specified') != 'Not specified':
                    st.write(f"**Preferred Skills:** {selected_job.get('preferred_skills')}")
        
        with col2:
            st.markdown("### ⚙️ Email Generation")
            
            # Option to customize portfolio links
            with st.expander("Portfolio Settings (Optional)"):
                use_custom_links = st.checkbox("Use custom portfolio links")
                custom_links = []
                
                if use_custom_links:
                    num_links = st.number_input("Number of custom links", min_value=1, max_value=5, value=1)
                    for i in range(num_links):
                        link = st.text_input(f"Portfolio Link {i+1}", key=f"link_{i}")
                        if link:
                            custom_links.append(link)
            
            # Generate email button
            generate_email_btn = st.button("🚀 Generate Cold Email", type="primary")
            
            if generate_email_btn:
                try:
                    with st.spinner("Generating personalized cold email..."):
                        # Get portfolio links
                        if use_custom_links and custom_links:
                            links = custom_links
                        else:
                            skills = selected_job.get('skills', [])
                            if isinstance(skills, str):
                                skills = [s.strip() for s in skills.split(',')]
                            links = portfolio.query_links(skills)
                        
                        # Generate email
                        email = llm.write_mail(selected_job, links)
                        
                        # Display the email
                        st.markdown("---")
                        st.subheader("📧 Generated Cold Email")
                        
                        # Create two columns for email display and actions
                        email_col1, email_col2 = st.columns([3, 1])
                        
                        with email_col1:
                            st.code(email, language='markdown')
                        
                        with email_col2:
                            st.markdown("### Actions")
                            
                            # Copy to clipboard button (using st.code with copy button)
                            if st.button("📋 Copy Email"):
                                st.success("Email copied to clipboard!")
                            
                            # Download as text file
                            st.download_button(
                                label="💾 Download Email",
                                data=email,
                                file_name=f"cold_email_{selected_job.get('role', 'job').replace(' ', '_').lower()}.txt",
                                mime="text/plain"
                            )
                            
                            # Regenerate button
                            if st.button("🔄 Regenerate Email"):
                                st.rerun()
                        
                        # Store generated email in session state for potential regeneration
                        if 'generated_emails' not in st.session_state:
                            st.session_state.generated_emails = {}
                        st.session_state.generated_emails[selected_job_index] = email
                        
                except Exception as e:
                    st.error(f"Error generating email: {e}")
        
        # Show summary of all jobs if multiple exist
        if len(st.session_state.jobs_data) > 1:
            st.markdown("---")
            st.subheader("📊 All Job Postings Summary")
            
            for i, job in enumerate(st.session_state.jobs_data):
                with st.expander(f"Job {i+1}: {job.get('role', 'Unknown Role')} at {job.get('company_name', 'Unknown Company')}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Experience:** {job.get('experience', 'Not specified')}")
                        st.write(f"**Skills:** {job.get('skills', 'Not specified')}")
                    with col_b:
                        st.write(f"**Requirements:** {job.get('requirements', 'Not specified')}")
                        if job.get('company_challenges') != 'Not specified':
                            st.write(f"**Challenges:** {job.get('company_challenges')}")

    # Sidebar with instructions and tips
    with st.sidebar:
        st.markdown("## 📖 How to Use")
        st.markdown("""
        1. **Enter URL**: Paste the job posting URL
        2. **Extract Jobs**: Click to find all job postings
        3. **Select Job**: Choose from dropdown if multiple jobs found
        4. **Customize**: Optionally add custom portfolio links
        5. **Generate**: Create your personalized cold email
        6. **Download**: Save or copy the generated email
        """)
        
        st.markdown("## 💡 Tips")
        st.markdown("""
        - Use career pages with multiple job listings
        - The AI will automatically match your skills to job requirements
        - Review the job details before generating email
        - Customize portfolio links for specific roles
        """)
        
        st.markdown("## 🔗 Supported Sites")
        st.markdown("""
        - Company career pages
        - Job board listings
        - Individual job posting URLs
        """)


if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(chain, clean_text)