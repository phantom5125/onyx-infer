from tools.modelscope_fetch import plan_ranges


def test_range_plan_resumes_and_covers_every_byte_once():
    total = 1024
    for existing in (0, 1, 255, 256, 257, 1023, 1024):
        ranges = plan_ranges(total, existing, chunk_size=256)
        assert sum(end - start for start, end in ranges) == total - existing
        assert ranges[0][0] == existing if ranges else True
        assert ranges[-1][1] == total if ranges else True
        for previous, current in zip(ranges, ranges[1:]):
            assert previous[1] == current[0]
        for start, end in ranges[1:]:
            assert start % 256 == 0
