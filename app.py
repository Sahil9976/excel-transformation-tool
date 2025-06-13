from flask import Flask, request, render_template, send_file, flash, redirect, url_for, jsonify
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Tuple, Any
import warnings
from werkzeug.utils import secure_filename
import tempfile
import shutil
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
TEMP_FOLDER = 'temp'

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, TEMP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Code 1: Formatting Logic
def process_formatting(input_path: str, output_path: str) -> None:
    """Reads an Excel file, cleans all sheets, and writes to a new Excel file"""
    xls = pd.ExcelFile(input_path)
    sheet_names = xls.sheet_names
    cleaned_sheets = {}

    def find_header_row(df):
        for i in range(min(10, len(df))):
            if df.iloc[i].notna().sum() > 2:
                return i
        return 0

    for sheet_name in sheet_names:
        df_raw = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
        header_row_index = find_header_row(df_raw)
        print(f"Processing {input_path} → Sheet '{sheet_name}' → Row {header_row_index + 1} as header")

        new_header = df_raw.iloc[header_row_index]
        df_clean = df_raw.iloc[header_row_index + 1:].copy()
        df_clean.columns = new_header
        df_clean.reset_index(drop=True, inplace=True)

        cleaned_sheets[sheet_name] = df_clean

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet, data in cleaned_sheets.items():
            data.to_excel(writer, sheet_name=sheet, index=False)

    print(f"✅ Cleaned and saved: {output_path}")

# Code 2: Excel Transformation Logic
class ExcelTransformationTool:
    def __init__(self):
        # Canonical headers as defined
        self.canonical_headers = [
            "State", "DeliveryHub", "AM Names", "AgentName", "FHRID", 
            "Grand Total", "Man Days", "Assigned", "Average", "Conversion", 
            "Bucket", "Cost deduction @ 0.20", "MG Payout", "Additional Del-1", 
            "Variable Payout-1", "Rate", "Cost Deduction", "After Deduction", 
            "Payout", "Fixed Amount", "Variable Payout-2", "LMA Deliveries", 
            "LMA Payout", "Chennai Rain Incentives", "Deduction", "Final payout", 
            "Remarks", "Doc- Status", "Period Start Date", "Period End Date"
        ]
        
        # Initialize sentence transformer model for semantic similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Precompute embeddings for canonical headers
        self.canonical_embeddings = self.model.encode(self.canonical_headers)
        
        # Define synonym mappings for better matching
        self.synonym_mappings = {
            "State": ["state", "location", "region", "province", "territory"],
            "DeliveryHub": ["hub", "delivery_hub", "distribution_center", "warehouse", "center"],
            "AM Names": ["am_name", "account_manager", "area_manager", "manager_name"],
            "AgentName": ["agent", "agent_name", "representative", "rep_name", "sales_agent"],
            "FHRID": ["fhr_id", "employee_id", "emp_id", "id", "worker_id"],
            "Grand Total": ["total", "grand_total", "sum", "total_amount", "overall_total"],
            "Man Days": ["mandays", "working_days", "work_days", "person_days"],
            "Assigned": ["assigned", "allocation", "designated", "allotted"],
            "Average": ["avg", "mean", "average_value", "typical"],
            "Conversion": ["conversion_rate", "convert", "success_rate"],
            "Bucket": ["category", "group", "classification", "segment"],
            "Cost deduction @ 0.20": ["cost_deduction", "deduction_20", "cost_reduction"],
            "MG Payout": ["mg_payout", "minimum_guarantee", "base_payout"],
            "Additional Del-1": ["additional_delivery", "extra_delivery", "bonus_delivery"],
            "Variable Payout-1": ["variable_payout", "performance_payout", "incentive"],
            "Rate": ["rate", "price", "cost_per_unit", "unit_rate"],
            "Cost Deduction": ["deduction", "cost_cut", "expense_reduction"], 
            "After Deduction": ["net_amount", "after_deduction", "final_amount"],
            "Payout": ["payment", "compensation", "remuneration", "salary"],
            "Fixed Amount": ["fixed_pay", "base_amount", "fixed_compensation"],
            "Variable Payout-2": ["variable_pay_2", "bonus_payout", "performance_bonus"],
            "LMA Deliveries": ["lma_delivery", "last_mile_delivery", "final_delivery"],
            "LMA Payout": ["lma_payment", "delivery_payout", "last_mile_pay"],
            "Chennai Rain Incentives": ["rain_incentive", "weather_bonus", "chennai_bonus"],
            "Deduction": ["deduction", "cut", "reduction", "penalty"],
            "Final payout": ["final_payment", "net_payout", "total_payout"],
            "Remarks": ["comments", "notes", "observations", "feedback"],
            "Doc- Status": ["document_status", "doc_status", "status", "document_state"],
            "Period Start Date": ["start_date", "period_start", "from_date", "begin_date"],
            "Period End Date": ["end_date", "period_end", "to_date", "finish_date"]
        }
    
    def preprocess_header(self, header: str) -> str:
        """Clean and preprocess header text for better matching"""
        if pd.isna(header):
            return ""
        
        # Convert to string and clean
        header = str(header).strip()
        
        # Remove special characters and normalize
        header = re.sub(r'[^\w\s]', ' ', header)
        header = re.sub(r'\s+', ' ', header)
        
        return header.lower()
    
    def exact_match(self, vendor_header: str) -> str:
        """Check for exact matches (case-insensitive)"""
        vendor_clean = self.preprocess_header(vendor_header)
        
        for canonical in self.canonical_headers:
            if vendor_clean == self.preprocess_header(canonical):
                return canonical
        
        return None
    
    def synonym_match(self, vendor_header: str) -> str:
        """Check for synonym matches"""
        vendor_clean = self.preprocess_header(vendor_header)
        
        for canonical, synonyms in self.synonym_mappings.items():
            if vendor_clean in [self.preprocess_header(syn) for syn in synonyms]:
                return canonical
        
        return None
    
    def semantic_match(self, vendor_header: str, threshold: float = 0.7) -> Tuple[str, float]:
        """Find semantic matches using sentence transformers"""
        if not vendor_header or pd.isna(vendor_header):
            return None, 0.0
        
        # Encode the vendor header
        vendor_embedding = self.model.encode([str(vendor_header)])
        
        # Calculate cosine similarity with all canonical headers
        similarities = cosine_similarity(vendor_embedding, self.canonical_embeddings)[0]
        
        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity >= threshold:
            return self.canonical_headers[best_match_idx], best_similarity
        
        return None, best_similarity
    
    def map_header(self, vendor_header: str) -> Tuple[str, str, float]:
        """Map vendor header to canonical header using multiple strategies"""
        if not vendor_header or pd.isna(vendor_header):
            return None, "empty", 0.0
        
        # Strategy 1: Exact match
        exact = self.exact_match(vendor_header)
        if exact:
            return exact, "exact", 1.0
        
        # Strategy 2: Synonym match
        synonym = self.synonym_match(vendor_header)
        if synonym:
            return synonym, "synonym", 0.95
        
        # Strategy 3: Semantic similarity
        semantic, similarity = self.semantic_match(vendor_header)
        if semantic:
            return semantic, "semantic", similarity
        
        return None, "no_match", 0.0
    
    def process_excel_file(self, file_path: str) -> pd.DataFrame:
        """Process an Excel file and transform it to canonical format"""
        print(f"Processing file: {file_path}")
        
        # Read all sheets from the Excel file
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        # Initialize result dataframe with canonical headers
        result_df = pd.DataFrame(columns=self.canonical_headers)
        
        mapping_report = []
        
        # Process each sheet
        for sheet_name, sheet_df in excel_data.items():
            print(f"\nProcessing sheet: {sheet_name}")
            
            # Skip empty sheets
            if sheet_df.empty:
                continue
            
            # Map each column in the sheet
            for vendor_col in sheet_df.columns:
                canonical_col, match_type, confidence = self.map_header(vendor_col)
                
                mapping_info = {
                    'sheet': sheet_name,
                    'vendor_header': vendor_col,
                    'canonical_header': canonical_col,
                    'match_type': match_type,
                    'confidence': confidence
                }
                mapping_report.append(mapping_info)
                
                if canonical_col:
                    print(f"  Mapped '{vendor_col}' -> '{canonical_col}' ({match_type}, {confidence:.2f})")
                    
                    # Append data to the result dataframe
                    vendor_data = sheet_df[vendor_col].dropna()
                    
                    if not vendor_data.empty:
                        # Create temporary dataframe for this column
                        temp_df = pd.DataFrame({canonical_col: vendor_data})
                        
                        # Concatenate with result dataframe
                        if canonical_col in result_df.columns:
                            # Append to existing column
                            current_data = result_df[canonical_col].dropna()
                            combined_data = pd.concat([current_data, vendor_data], ignore_index=True)
                            
                            # Ensure result_df is large enough
                            if len(combined_data) > len(result_df):
                                result_df = result_df.reindex(range(len(combined_data)))
                            
                            result_df[canonical_col] = combined_data
                        else:
                            # First time adding this column
                            if len(vendor_data) > len(result_df):
                                result_df = result_df.reindex(range(len(vendor_data)))
                            result_df[canonical_col] = vendor_data
                else:
                    print(f"  No match found for '{vendor_col}'")
        
        return result_df, mapping_report
    
    def save_results(self, result_df: pd.DataFrame, mapping_report: List[Dict], 
                    output_path: str, report_path: str = None):
        """Save the transformed data and mapping report"""
        
        # Save the transformed data
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            result_df.to_excel(writer, sheet_name='Transformed_Data', index=False)
        
        print(f"\nTransformed data saved to: {output_path}")
        
        # Save mapping report if requested
        if report_path:
            report_df = pd.DataFrame(mapping_report)
            report_df.to_excel(report_path, index=False)
            print(f"Mapping report saved to: {report_path}")
        
        return result_df
    
    def get_mapping_statistics(self, mapping_report: List[Dict]) -> Dict:
        """Generate statistics about the mapping process"""
        total_columns = len(mapping_report)
        matched_columns = len([m for m in mapping_report if m['canonical_header'] is not None])
        
        match_types = {}
        for report in mapping_report:
            match_type = report['match_type']
            match_types[match_type] = match_types.get(match_type, 0) + 1
        
        avg_confidence = np.mean([m['confidence'] for m in mapping_report if m['confidence'] > 0])
        
        return {
            'total_columns': total_columns,
            'matched_columns': matched_columns,
            'match_rate': matched_columns / total_columns if total_columns > 0 else 0,
            'match_types': match_types,
            'average_confidence': avg_confidence
        }

# Initialize the transformer
transformer = ExcelTransformationTool()

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Step 1: Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            
            upload_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(upload_path)
            
            # Step 2: Format the file (Code 1)
            formatted_filename = f"formatted_{unique_filename}"
            formatted_path = os.path.join(TEMP_FOLDER, formatted_filename)
            process_formatting(upload_path, formatted_path)
            
            # Step 3: Transform the file (Code 2)
            final_filename = f"final_{unique_filename}"
            final_path = os.path.join(PROCESSED_FOLDER, final_filename)
            report_filename = f"report_{unique_filename}"
            report_path = os.path.join(PROCESSED_FOLDER, report_filename)
            
            result_df, mapping_report = transformer.process_excel_file(formatted_path)
            transformer.save_results(result_df, mapping_report, final_path, report_path)
            
            # Get statistics
            stats = transformer.get_mapping_statistics(mapping_report)
            
            # Clean up temporary files
            os.remove(upload_path)
            os.remove(formatted_path)
            
            return render_template('results.html', 
                                 stats=stats, 
                                 final_file=final_filename,
                                 report_file=report_filename,
                                 mapping_report=mapping_report[:10])  # Show first 10 mappings
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload .xlsx or .xls files only.')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(os.path.join(PROCESSED_FOLDER, filename), as_attachment=True)
    except Exception as e:
        flash('File not found')
        return redirect(url_for('index'))

@app.route('/api/progress')
def get_progress():
    # This is a placeholder for progress tracking
    # You can implement real-time progress updates here
    return jsonify({'progress': 100, 'status': 'completed'})

if __name__ == '__main__':
    app.run(debug=True)