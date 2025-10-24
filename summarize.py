import os
import re
import dotenv
import threading
import pandas as pd
from tqdm import tqdm
from together import Together 
from concurrent.futures import ThreadPoolExecutor, as_completed


def summarize(path): 
    dotenv.load_dotenv()
    df = pd.read_csv(path)
    # Thread-local storage for Together client
    thread_local = threading.local()

    def get_client():
        """Get or create a Together client for the current thread"""
        if not hasattr(thread_local, 'client'):
            thread_local.client = Together()
        return thread_local.client

    def extract_summary(response_text):
        """Extract the actual summary from response, removing thinking tags if present"""
        if response_text is None:
            return ""
        
        # Remove thinking tags and content
        # Pattern: <think>...</think> at the start
        cleaned = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
        return cleaned.strip()

    def process_row(row):
        """Process a single row to generate all 4 summaries"""
        try:
            document = row['document']
            topic_prompt = row['summarization_prompt']
            base_prompt = "Summarize the following document: " 
            overall_topic_prompt = "{} \n\n Document: {}".format(topic_prompt, document)
            overall_base_prompt = "{} \n\n Document: {}".format(base_prompt, document)
            
            client = get_client()

            # Thinking with base prompt
            thinking_base_response = client.chat.completions.create( 
                model="Qwen/Qwen3-235B-A22B-fp8-tput",
                messages=[ 
                    {
                        "role": "user", 
                        "content": overall_base_prompt,
                        "enable_thinking": True
                    } 
                ] 
            )  
            thinking_base_summary = extract_summary(thinking_base_response.choices[0].message.content)

            # Thinking with topic prompt
            thinking_topic_response = client.chat.completions.create( 
                model="Qwen/Qwen3-235B-A22B-fp8-tput",
                messages=[ 
                    {
                        "role": "user", 
                        "content": overall_topic_prompt,
                        "enable_thinking": True
                    } 
                ] 
            )  
            thinking_topic_summary = extract_summary(thinking_topic_response.choices[0].message.content)

            # Non-thinking with base prompt
            non_thinking_base_response = client.chat.completions.create( 
                model="Qwen/Qwen3-235B-A22B-fp8-tput",
                messages=[ 
                    {
                        "role": "user", 
                        "content": overall_base_prompt,
                        "enable_thinking": False
                    } 
                ] 
            )  
            non_thinking_base_summary = non_thinking_base_response.choices[0].message.content

            # Non-thinking with topic prompt
            non_thinking_topic_response = client.chat.completions.create( 
                model="Qwen/Qwen3-235B-A22B-fp8-tput",
                messages=[ 
                    {
                        "role": "user", 
                        "content": overall_topic_prompt,
                        "enable_thinking": False
                    } 
                ] 
            )  
            non_thinking_topic_summary = non_thinking_topic_response.choices[0].message.content
            
            return {
                'id': row['id'],
                'thinking_base_summary': thinking_base_summary, 
                'thinking_topic_summary': thinking_topic_summary, 
                'non_thinking_base_summary': non_thinking_base_summary, 
                'non_thinking_topic_summary': non_thinking_topic_summary,
                'success': True
            }
        except Exception as e:
            return {
                'id': row['id'],
                'error': str(e),
                'success': False
            }

    # Load existing results if available
    output_path = 'data/regular_generated_summaries.csv'
    processed_ids = set()

    if os.path.exists(output_path):
        print(f"Loading existing results from {output_path}...")
        try:
            existing_df = pd.read_csv(output_path, index_col=0)
            processed_ids = set(existing_df.index)
            print(f"✓ Found {len(processed_ids)} already processed documents")
        except Exception as e:
            print(f"Warning: Could not load existing CSV ({e}), starting fresh")
            processed_ids = set()

    # Filter out already processed rows
    rows_to_process = [(idx, row) for idx, row in df.iterrows() if row['id'] not in processed_ids]
    print(f"Processing {len(rows_to_process)} remaining documents (out of {len(df)} total)")

    if len(rows_to_process) == 0:
        print("All documents already processed!")
    else:
        # Process with thread pool
        max_workers = 10  # Adjust based on API rate limits
        
        print(f"Using {max_workers} parallel workers...")
        
        # Lock for thread-safe file writing
        write_lock = threading.Lock()
        
        # Track all results for final save
        all_results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            futures = {
                executor.submit(process_row, row): (idx, row['id']) 
                for idx, row in rows_to_process
            }
            
            # Process completed jobs with progress bar
            success_count = 0
            error_count = 0
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating summaries"):
                result = future.result()
                
                if result['success']:
                    all_results[result['id']] = {
                        'thinking_base_summary': result['thinking_base_summary'],
                        'thinking_topic_summary': result['thinking_topic_summary'],
                        'non_thinking_base_summary': result['non_thinking_base_summary'],
                        'non_thinking_topic_summary': result['non_thinking_topic_summary']
                    }
                    success_count += 1
                    
                    # Save every 50 results
                    if success_count % 5 == 0:
                        with write_lock:
                            results_df = pd.DataFrame.from_dict(all_results, orient='index')
                            results_df.to_csv(output_path)
                            print(f"\n✓ Checkpoint: Saved {len(all_results)} results")
                else:
                    print(f"\nError processing {result['id']}: {result['error']}")
                    error_count += 1
        
        # Final save
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.to_csv(output_path)
        
        print(f"\n✓ Completed: {success_count} successful, {error_count} errors")
        print(f"✓ Final results saved to {output_path}")


if __name__ == "__main__":
    input_path = 'data/train_combined_with_prompts.csv'
    summarize(input_path)