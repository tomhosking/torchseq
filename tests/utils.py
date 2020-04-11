def slow(test_case):
    """
    Decorator marking a test as slow.
    Slow tests are skipped by default. Set the RUN_SLOW environment variable
    to a truthy value to run them.
    """
    if not _run_slow_tests:
        test_case = unittest.skip("test is slow")(test_case)
    return test_case