from copilot.core.parser import parse_file

def test_parse():
    meta, utts = parse_file('data/1_demo_call.txt')
    assert meta.call_id.startswith('1_demo_call')
    assert len(utts) > 0
    # check timestamp format
    assert all(u.start_sec >= 0 and u.end_sec >= u.start_sec for u in utts)
