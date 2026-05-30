import pandas as pd
from pathlib import Path


def analyze_clusters() -> None:
    """
    Analyze and print the number of data points per cluster for all analysis types.
    """
    analysis_dir = Path(__file__).parent / "mace_analysis"
    
    # Define the files to analyze
    files_to_analyze = [
        "protein_counts_mace_pic50.csv",
        "protein_counts_mace_pki.csv",
        "protein_counts_pocket_pic50.csv",
        "protein_counts_pocket_pki.csv",
        "protein_counts_protein_pic50.csv",
        "protein_counts_protein_pki.csv",
    ]
    
    results = {}
    
    for file in files_to_analyze:
        file_path = analysis_dir / file
        
        if not file_path.exists():
            print(f"Warning: {file} not found")
            continue
            
        df = pd.read_csv(file_path)
        
        # Group by cluster and sum the counts
        cluster_counts = df.groupby('Cluster')['Count'].sum().sort_index()
        total = cluster_counts.sum()
        
        # Extract analysis type and dataset
        parts = file.replace("protein_counts_", "").replace(".csv", "").split("_")
        analysis_type = parts[0].upper()
        dataset = parts[1].upper()
        
        key = f"{analysis_type} - {dataset}"
        results[key] = {
            'cluster_counts': cluster_counts,
            'total': total,
            'num_clusters': len(cluster_counts)
        }
        
        print(f"\n{'='*60}")
        print(f"{key}")
        print(f"{'='*60}")
        print(f"Number of clusters: {len(cluster_counts)}")
        print("\nData points per cluster:")
        for cluster, count in cluster_counts.items():
            print(f"  Cluster {cluster}: {count}")
        print(f"\nTotal data points: {total}")
    
    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Analysis Type':<20} {'Dataset':<10} {'Clusters':<12} {'Total Points':<15}")
    print(f"{'-'*80}")
    
    for key, data in results.items():
        parts = key.split(" - ")
        analysis = parts[0]
        dataset = parts[1]
        print(f"{analysis:<20} {dataset:<10} {data['num_clusters']:<12} {data['total']:<15}")

if __name__ == "__main__":
    analyze_clusters()
