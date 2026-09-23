"""
ShoppingBench Integrated Runner

This script runs shopping agent inference for different levels.

Usage:
    python run.py --model qwen-plus --level 1 --workers 40
"""

import argparse
import os
import sys
import time
from pathlib import Path
SHOPPING_AGENT_BACKEND = os.getenv('SHOPPING_AGENT_BACKEND', 'default')
if SHOPPING_AGENT_BACKEND == 'react':
    from reactAgent.shopping_agent import run_agent_inference
    from reactAgent.prompts import prompt_lib
elif SHOPPING_AGENT_BACKEND == 'gurobi':
    from GurobiAgent.shopping_agent import prompt_lib, run_agent_inference
else:
    from agent.shopping_agent import run_agent_inference
    from agent.prompts import prompt_lib

sys.path.insert(0, str(Path(__file__).parent))
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run ShoppingBench agent inference'
    )
    
    # Model configuration
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (default: from SHOPPING_AGENT_MODEL env or qwen-plus)')
    
    # Level configuration
    parser.add_argument('--level', type=int, default=1, choices=[1, 2, 3],
                       help='Task level, determines input file and system prompt (default: 1)')
    
    # Execution configuration
    parser.add_argument('--workers', type=int, default=5,
                       help='Number of concurrent workers (default: 5)')
    parser.add_argument('--max-llm-calls', type=int, default=400,
                       help='Maximum LLM calls per sample (default: 400)')
    parser.add_argument('--rerun-ids', type=int, nargs='+', default=None,
                       help='Only rerun the specified sample IDs')
    
    parser.add_argument('--test-data', type=Path, default=None,
                       help='Query JSON file (default: data/level_{level}_query_meta.json)')

    # Database configuration (for concurrent run isolation)
    parser.add_argument('--database-dir', type=str, default=None,
                       help='Path to database directory (default: database/). Use unique paths to allow concurrent runs.')
    
    # Advanced options
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    return args

def setup_paths(args):
    """Setup paths for input/output directories"""
    base_dir = Path(__file__).parent
    
    # Test data path - based on level
    args.test_data = args.test_data or base_dir / 'data' / f'level_{args.level}_query_meta.json'
    
    if not args.test_data.exists():
        raise FileNotFoundError(f"Test data file not found: {args.test_data}")
    
    # Database path - use provided path or default to 'database/'
    # Using --database-dir allows concurrent runs with isolated databases
    if args.database_dir:
        # Support both relative and absolute paths
        db_path = Path(args.database_dir)
        if not db_path.is_absolute():
            db_path = base_dir / db_path
        args.database_dir = db_path
    else:
        args.database_dir = base_dir / 'database'
    
    if not args.database_dir.exists():
        raise FileNotFoundError(f"Database directory not found: {args.database_dir}")
    
    # Tool schema path
    args.tool_schema_path = base_dir / 'tools' / 'shopping_tool_schema.json'
    
    if not args.tool_schema_path.exists():
        raise FileNotFoundError(f"Tool schema file not found: {args.tool_schema_path}")
    
    return args
def print_config(args):
    """Print configuration summary"""
    print("=" * 80)
    print("ShoppingBench Integrated Runner")
    print("=" * 80)
    print(f"Backend:            {SHOPPING_AGENT_BACKEND}")
    print(f"Model:              {args.model}")
    print(f"Level:              {args.level}")
    print(f"Workers:            {args.workers}")
    print(f"Max LLM calls:      {args.max_llm_calls}")
    print(f"Test data:          {args.test_data}")
    print(f"Database directory: {args.database_dir}")
    print(f"Tool schema:        {args.tool_schema_path}")
    print("=" * 80)
    print()

def run_step_inference(args):
    """Run agent inference to generate trajectories"""
    
    # Get system prompt based on level
    system_prompt_attr = f"SYSTEM_PROMPT_level{args.level}"
    if not hasattr(prompt_lib, system_prompt_attr):
        print(f"❌ System Prompt not found: {system_prompt_attr} in utils.prompts")
        return False, None
    system_prompt = getattr(prompt_lib, system_prompt_attr)
    
    
    start_time = time.time()
    
    try:
        inference_kwargs = dict(
            model=args.model,
            test_data_path=args.test_data,
            database_dir=args.database_dir,
            tool_schema_path=args.tool_schema_path,
            system_prompt=system_prompt,
            workers=args.workers,
            max_llm_calls=args.max_llm_calls,
            rerun_ids=args.rerun_ids,
        )
        if SHOPPING_AGENT_BACKEND != 'react':
            inference_kwargs['level'] = args.level
        result = run_agent_inference(**inference_kwargs)
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Inference completed in {elapsed:.2f}s")
        print(f"   Total samples: {result['total']}")
        print(f"   Success: {result['success']}")
        print(f"   Failed: {result['failed']}")
        
        return True, {
            'total': result['total'],
            'success': result['success'],
            'failed': result['failed']
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Inference failed after {elapsed:.2f}s: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return False, None


def main():
    """Main execution function"""
    args = parse_args()
    
    # Set default model if not provided
    if args.model is None:
        args.model = os.getenv('SHOPPING_AGENT_MODEL', 'qwen-plus')
    
    # Setup paths
    args = setup_paths(args)
    
    # Print configuration
    print_config(args)
    
    overall_start_time = time.time()
    
    # Run inference
    success, inference_results = run_step_inference(args)
    
    if not success:
        print("\n❌ Inference failed")
        sys.exit(1)
    
    # Print final summary
    overall_elapsed = time.time() - overall_start_time
    print(f"\nTotal time: {overall_elapsed:.2f}s ({overall_elapsed/60:.1f} minutes)")
    
    print("\n✅ Pipeline completed successfully!")
if __name__ == '__main__':
    main()
