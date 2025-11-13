import pandas as pd
from tqdm import tqdm
import streamlit as st
import io
from datetime import datetime


# Function to handle prefixes that would create too many columns
def limit_prefix_columns(df, prefixes_to_limit):
    """
    For specified prefixes, keep only the _Value column (renamed to just the prefix)
    and remove all other related columns to prevent column explosion.

    Args:
        df: Input DataFrame
        prefixes_to_limit: Set of prefix names to limit

    Returns:
        Modified DataFrame with limited prefixes cleaned up
    """
    print(f"\nLimiting column explosion for prefixes: {prefixes_to_limit}")

    # Track which prefixes were actually limited
    columns_to_remove = set()
    columns_to_rename = {}

    for prefix in prefixes_to_limit:
        header_col = f"{prefix}_Header"
        value_col = f"{prefix}_Value"

        if header_col in df.columns:
            print(f"  Limiting {prefix}: keeping only {value_col} (renamed to {prefix})")

            # Rename _Value column to just the prefix name
            if value_col in df.columns:
                columns_to_rename[value_col] = prefix

            # Mark all other related columns for removal
            related_cols = [
                f"{prefix}_Header",
                f"{prefix}_Header Code",
                f"{prefix}_Value Code",
                f"{prefix}_Unit",
                f"{prefix}_Unit Code"
            ]
            for col in related_cols:
                if col in df.columns:
                    columns_to_remove.add(col)

    # Rename the _Value columns
    if columns_to_rename:
        df = df.rename(columns=columns_to_rename)
        print(f"  Renamed columns: {columns_to_rename}")

    # Remove the other related columns
    if columns_to_remove:
        df = df.drop(columns=list(columns_to_remove))
        print(f"  Removed {len(columns_to_remove)} columns")

    return df

# Transform the DataFrame
def transform_dataframe(df):
    """
    Transform DataFrame columns based on three patterns:
    1. _Header, _Value, _Unit columns -> Create columns for each header value
    2. _Enum Value columns -> Create single column without _Enum Value suffix
    3. Regular columns -> Keep as-is
    """

    # Identify column patterns
    all_columns = list(df.columns)

    # Find all prefixes with _Header pattern
    header_prefixes = set()
    for col in all_columns:
        if col.endswith('_Header'):
            prefix = col.replace('_Header', '')
            # Check if corresponding _Value and _Unit columns exist
            if f"{prefix}_Value" in all_columns and f"{prefix}_Unit" in all_columns:
                header_prefixes.add(prefix)

    # Find all prefixes with _Enum Value pattern
    enum_prefixes = set()
    for col in all_columns:
        if col.endswith('_Enum Value'):
            prefix = col.replace('_Enum Value', '')
            enum_prefixes.add(prefix)

    # Identify columns that are part of patterns
    pattern_columns = set()
    for prefix in header_prefixes:
        pattern_columns.add(f"{prefix}_Header")
        pattern_columns.add(f"{prefix}_Header Code")
        pattern_columns.add(f"{prefix}_Value")
        pattern_columns.add(f"{prefix}_Value Code")
        pattern_columns.add(f"{prefix}_Unit")
        pattern_columns.add(f"{prefix}_Unit Code")

    for prefix in enum_prefixes:
        pattern_columns.add(f"{prefix}_Enum Value")
        pattern_columns.add(f"{prefix}_Enum Code")

    # Keep only columns that exist in the dataframe
    pattern_columns = pattern_columns.intersection(all_columns)

    # Regular columns are those not in any pattern
    regular_columns = [col for col in all_columns if col not in pattern_columns]

    print(f"\nFound {len(header_prefixes)} prefixes with _Header pattern: {header_prefixes}")
    print(f"Found {len(enum_prefixes)} prefixes with _Enum Value pattern: {enum_prefixes}")
    print(f"Found {len(regular_columns)} regular columns")

    # Start building the new DataFrame
    new_df = df[regular_columns].copy()

    # Process _Header/_Value/_Unit patterns - collect all columns first, then concat once
    if header_prefixes:
        header_columns = {}

        for prefix in tqdm(header_prefixes, desc="Processing _Header patterns", smoothing=0.0):
            header_col = f"{prefix}_Header"
            value_col = f"{prefix}_Value"
            unit_col = f"{prefix}_Unit"

            # Get unique header values (including NaN for empty headers)
            unique_headers = df[header_col].unique()  # Don't drop NaN - we want to handle empty headers

            # Create a column for each unique header value using vectorized operations
            for header_value in unique_headers:
                # Determine the column name based on whether header is empty
                if pd.isna(header_value):
                    # If header is empty/NaN, just use the prefix as column name
                    new_col_name = prefix
                else:
                    # Convert header_value to int if it's a whole number float
                    try:
                        numeric_header = pd.to_numeric(header_value, errors='coerce')
                        if pd.notna(numeric_header) and numeric_header == round(numeric_header):
                            header_value = int(numeric_header)
                    except:
                        pass  # Keep original value if conversion fails

                    new_col_name = f"{prefix}; {header_value}"

                # Create mask for rows where header matches (handle NaN properly)
                if pd.isna(header_value):
                    mask = df[header_col].isna()
                else:
                    mask = df[header_col] == header_value

                # Get value series and convert floats to ints when appropriate
                value_series = df.loc[mask, value_col].copy()

                # Convert float to int if it's a whole number
                # First, try to convert to numeric (will return NaN for non-numeric strings)
                numeric_values = pd.to_numeric(value_series, errors='coerce')
                is_numeric = numeric_values.notna()
                is_whole = is_numeric & (numeric_values == numeric_values.round())

                # Create a new series for string representation
                value_series_str = pd.Series(index=value_series.index, dtype=str)

                # For whole numbers, convert to int then string
                value_series_str.loc[is_whole] = numeric_values.loc[is_whole].astype(int).astype(str)
                # For non-whole numbers, convert directly to string
                value_series_str.loc[is_numeric & ~is_whole] = numeric_values.loc[is_numeric & ~is_whole].astype(str)
                # For non-numeric values, use original string
                value_series_str.loc[~is_numeric] = value_series.loc[~is_numeric].astype(str)
                # Replace 'nan' with empty string
                value_series_str = value_series_str.replace('nan', '').replace('<NA>', '')

                unit_series = df.loc[mask, unit_col].fillna('').astype(str)
                # Replace 'nan' and 'No unit' with empty string
                unit_series = unit_series.replace('nan', '').replace('No unit', '')

                # Initialize combined series with None for ALL rows
                combined = pd.Series(None, index=df.index, dtype=object)

                # Where both value and unit exist
                both_exist = (value_series_str != '') & (unit_series != '')
                combined.loc[mask & both_exist] = value_series_str.loc[both_exist] + ' ' + unit_series.loc[both_exist]

                # Where only value exists
                only_value = (value_series_str != '') & (unit_series == '')
                combined.loc[mask & only_value] = value_series_str.loc[only_value]

                # Store in dictionary instead of adding to DataFrame
                header_columns[new_col_name] = combined

        # Add all header columns at once
        header_df = pd.DataFrame(header_columns, index=df.index)
        new_df = pd.concat([new_df, header_df], axis=1)

    # Process _Enum Value patterns - collect all columns first, then concat once
    if enum_prefixes:
        enum_columns = {}
        for prefix in tqdm(enum_prefixes, desc="Processing _Enum Value patterns", smoothing=0.0):
            enum_col = f"{prefix}_Enum Value"
            new_col_name = prefix
            enum_columns[new_col_name] = df[enum_col]

        # Add all enum columns at once
        enum_df = pd.DataFrame(enum_columns, index=df.index)
        new_df = pd.concat([new_df, enum_df], axis=1)

    return new_df
def consolidate_by_product(df):
    """
    Consolidate multiple rows per Product ID into a single row.
    Master/Slave rows are combined by concatenating values with semicolon separator.
    """
    print("\nConsolidating rows by Product ID...")

    # Check if Product ID column exists
    product_id_col = "Product ID (ProductId)"
    if product_id_col not in df.columns:
        print(f"Warning: {product_id_col} column not found. Skipping consolidation.")
        return df

    print(f"Processing {len(df.columns)} columns for {df[product_id_col].nunique()} unique products...")

    # Group by Product ID
    grouped = df.groupby(product_id_col, sort=False)

    # Strategy: Use built-in agg functions instead of custom lambda
    # 1. Get first non-null value for each column
    print("Getting first non-null value for all columns...")
    result_df = grouped.first()

    # 2. Find groups that have multiple different non-null values per column
    # For those, we need to concatenate with semicolons
    print("Identifying columns with multiple values per product...")

    # Count unique non-null values per group per column
    nunique = grouped.nunique()

    # Find which (product, column) combinations have multiple values
    needs_concat = nunique > 1

    if needs_concat.any().any():
        total_concat_needed = needs_concat.sum().sum()
        print(f"Found {total_concat_needed} column-product combinations that need concatenation")

        # Only process columns that have at least one multi-value case
        cols_to_concat = needs_concat.any()[needs_concat.any()].index.tolist()

        print(f"Processing {len(cols_to_concat)} columns that need value concatenation...")

        for col in tqdm(cols_to_concat, desc="Concatenating multi-value columns", smoothing=0.0):
            # For this column, get mask of products that need concatenation
            products_to_concat = needs_concat[col][needs_concat[col]].index

            # Only concatenate for those specific products
            for product_id in products_to_concat:
                values = df.loc[df[product_id_col] == product_id, col].dropna().unique()
                if len(values) > 1:
                    result_df.loc[product_id, col] = '; '.join(str(v) for v in values)

    result_df = result_df.reset_index()

    print(f"Consolidated from {len(df)} rows to {len(result_df)} rows")

    return result_df

def process_and_save_dataframe(df):
    """
    Process a dataframe through the complete transformation pipeline:
    1. Limit columns for specified prefixes
    2. Transform dataframe structure
    3. Consolidate rows by Product ID
    4. Save to Excel

    Args:
        df: Input DataFrame to process

    Returns:
        DataFrame: The consolidated DataFrame
    """
    # Transform the dataframe
    print("\n" + "="*80)
    print("Starting transformation...")
    print("="*80)

    # Limit columns for prefixes that would create too many columns
    prefixes_to_limit = {'SimplifiedScip', 'Scip', 'Eti9LogAtt', 'Eti8LogAtt', 'Eti7LogAtt'}
    df_limited = limit_prefix_columns(df, prefixes_to_limit)

    df_transformed = transform_dataframe(df_limited)

    # Consolidate rows by Product ID
    print("\n" + "="*80)
    print("Consolidating rows by Product ID...")
    print("="*80)

    df_consolidated = consolidate_by_product(df_transformed)

    print("\n" + "="*80)
    print("Transformation complete!")
    print("="*80)
    print(f"\nFinal DataFrame has {len(df_consolidated)} rows and {len(df_consolidated.columns)} columns")

    print("Done!")

    return df_consolidated


def main():
    """
    Main Streamlit application for processing syndication data
    """
    st.set_page_config(
        page_title="Syndication to PowerText Converter",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Syndication to PowerText Input Converter")
    st.markdown("""
    Upload your syndication Excel file to transform it into PowerText input format.

    The tool will:
    1. Limit column explosion for specified prefixes
    2. Transform dataframe structure (Header/Value/Unit patterns)
    3. Consolidate rows by Product ID
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload your syndication Excel file"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            with st.spinner("Reading Excel file..."):
                df_input = pd.read_excel(uploaded_file)

            st.success(f"✅ File loaded successfully: {len(df_input)} rows, {len(df_input.columns)} columns")

            # Show input preview
            with st.expander("Preview Input Data (first 10 rows)"):
                st.dataframe(df_input.head(10))

            # Automatically process the data
            status_placeholder = st.empty()

            try:
                status_placeholder.info("🔄 Processing data... This may take a few minutes.")

                df_consolidated = process_and_save_dataframe(df_input)

                status_placeholder.success(f"✅ Processing complete! Result: {len(df_consolidated)} rows, {len(df_consolidated.columns)} columns")

                # Show output preview
                with st.expander("Preview Output Data (first 10 rows)"):
                    st.dataframe(df_consolidated.head(10))

                # Prepare download
                with st.spinner("Preparing Excel file for download..."):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"syndication_transformed_{timestamp}.xlsx"

                    # Convert to Excel in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_consolidated.to_excel(writer, index=False, sheet_name='Transformed Data')
                    output.seek(0)

                # Download button
                st.download_button(
                    label="⬇️ Download Transformed Excel",
                    data=output,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                status_placeholder.error(f"❌ Error processing data: {str(e)}")
                st.exception(e)

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.exception(e)

    else:
        st.info("👆 Please upload an Excel file to get started")


if __name__ == "__main__":
    main()
