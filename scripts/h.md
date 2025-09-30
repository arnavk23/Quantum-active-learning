Generating comprehensive benchmark analysis...

❌ Benchmarking failed: 'RF Uncertainty'
Traceback (most recent call last):
  File "/home/arnav/Downloads/camp/scripts/benchmark.py", line 1138, in main
    results = benchmark.run_comprehensive_benchmark()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/arnav/Downloads/camp/scripts/benchmark.py", line 608, in run_comprehensive_benchmark
    self._generate_comprehensive_analysis(all_results)
  File "/home/arnav/Downloads/camp/scripts/benchmark.py", line 1025, in _generate_comprehensive_analysis
    self._generate_benchmark_report(all_results, method_ranks)
  File "/home/arnav/Downloads/camp/scripts/benchmark.py", line 1081, in _generate_benchmark_report
    category = self.method_descriptions[method]['category']
               ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
KeyError: 'RF Uncertainty'
(.venv) arnav@kapoor:~/Downloads/camp$ 