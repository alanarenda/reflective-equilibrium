import json
import pandas as pd
from datasets import load_dataset


def prepare_govreport_data():
    # Load both datasets
    print("Loading datasets...")
    ds_qs = load_dataset("launch/gov_report_qs", "document")
    ds_gr = load_dataset("launch/gov_report", trust_remote_code=True)

    # Convert gov_report_qs splits to pandas
    train_qs = ds_qs["train"].to_pandas()
    val_qs = ds_qs["validation"].to_pandas()
    test_qs = ds_qs["test"].to_pandas()

    # Convert gov_report splits to pandas and combine ALL splits
    train_gr = ds_gr["train"].to_pandas()
    val_gr = ds_gr["validation"].to_pandas()
    test_gr = ds_gr["test"].to_pandas()
    all_gr = pd.concat([train_gr, val_gr, test_gr], ignore_index=True)

    # Join each split of gov_report_qs against the full gov_report dataset
    train_combined = train_qs.merge(all_gr, left_on='doc_id', right_on='id', how='inner')
    val_combined = val_qs.merge(all_gr, left_on='doc_id', right_on='id', how='inner')
    test_combined = test_qs.merge(all_gr, left_on='doc_id', right_on='id', how='inner')
    
    with open('train_combined.csv', 'w') as f:
        train_combined.to_csv(f, index=False)

    with open('val_combined.csv', 'w') as f:
        val_combined.to_csv(f, index=False) 

    with open('test_combined.csv', 'w') as f:
        test_combined.to_csv(f, index=False)    

    def build_question_hierarchy(qa_pairs):
        """Build hierarchical dictionary with paragraph-relative parent indices"""
        questions = list(qa_pairs['question'])
        summaries = list(qa_pairs['summary'])
        parent_indices = qa_pairs['parent_pair_index']
        paragraph_indices = qa_pairs['summary_paragraph_index']
        
        # Group questions by paragraph
        paragraph_groups = {}
        for i in range(len(questions)):
            para_idx = int(paragraph_indices[i])
            if para_idx not in paragraph_groups:
                paragraph_groups[para_idx] = []
            paragraph_groups[para_idx].append(i)
        
        hierarchy = {}
        
        # Create nodes with corrected parent indices
        for i in range(len(questions)):
            para_idx = int(paragraph_indices[i])
            local_parent_idx = int(parent_indices[i])
                
            if local_parent_idx == -1:
                global_parent_idx = None
            else:
                # Map local index within paragraph to global index
                para_questions = paragraph_groups[para_idx]
                if 0 <= local_parent_idx < len(para_questions):
                    global_parent_idx = para_questions[local_parent_idx]
                else:
                    global_parent_idx = None
            
            hierarchy[i] = {
                'question': questions[i],
                'summary': summaries[i],
                'paragraph': para_idx,
                'parent': global_parent_idx,
                'children': []
            }
        
        # Link children to parents
        root_nodes = []
        for i in range(len(questions)):
            parent_idx = hierarchy[i]['parent']
            if parent_idx is None:
                root_nodes.append(i)
            else:
                if parent_idx in hierarchy:
                    hierarchy[parent_idx]['children'].append(i)
        
        return {
            'hierarchy': hierarchy,
            'root_nodes': root_nodes,
            'paragraph_groups': paragraph_groups
        }

    def hierarchy_to_nested_dict(hierarchy_data):
        """Convert flat hierarchy to nested dictionary structure"""
        hierarchy = hierarchy_data['hierarchy']
        root_nodes = hierarchy_data['root_nodes']
        
        def build_node(node_idx):
            node = hierarchy[node_idx]
            nested_node = {
                'index': node_idx,
                'question': node['question'],
                'summary': node['summary'],
                'paragraph': node['paragraph'],
                'children': []
            }
            for child_idx in node['children']:
                nested_node['children'].append(build_node(child_idx))
            return nested_node
        
        nested_hierarchy = []
        for root_idx in root_nodes:
            nested_hierarchy.append(build_node(root_idx))
        
        return nested_hierarchy

    # Build hierarchies for all documents
    all_nested_hierarchies = {}

    print("Building nested hierarchies for all documents...")
    for idx, row in train_combined.iterrows():
        doc_id = row['doc_id']
        qa_pairs = row['question_summary_pairs']
        
        hierarchy = build_question_hierarchy(qa_pairs)
        nested = hierarchy_to_nested_dict(hierarchy)
        all_nested_hierarchies[doc_id] = nested

    print(f"✓ Built hierarchies for {len(all_nested_hierarchies)} documents")

    # Save to JSON
    with open('question_hierarchies_nested.json', 'w') as f:
        json.dump(all_nested_hierarchies, f, indent=2)
        
    print("✓ Saved to question_hierarchies_nested.json")
    

if __name__ == "__main__":
    prepare_govreport_data()