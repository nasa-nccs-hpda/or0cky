"""D187: print the figures of a hybrid run json (d187_coupled_step.py) for the ledger entry.  python d187_summary.py HYBDIR DATE"""
import json, sys
d = json.load(open(f'{sys.argv[1]}/{sys.argv[2]}_hyb.json'))
print(d['seconds'])
r = d['runs']
for x in r:
    print(x['label'], 'wall', round(x['wall_seconds'], 1), 'phase1', round(x['phase1_seconds'], 1), 'surf', round(x['surface_half_seconds'], 1), 'jit', x['jit_executions_total'],
          'eager', x['eager_primitive_dispatches_total'], x['flags'], x.get('end_state_bitwise_equal_to_cold_run'))
t = r[-1] if len(r) < 3 else r[2]
print(t['non_jax_text'])
print(t['jit_and_eager_by_phase'])
print('totals', t['counters_totals'])
for k, v in t['counters_by_phase'].items():
    print(' ', k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if b})
print('libimf', {k: {a: round(b, 3) if isinstance(b, float) else b for a, b in v.items()} for k, v in t['libimf_callbacks'].items() if k == 'total'})
print('pole', t['pole_callbacks'], 'qus', t['qus_callbacks'], 'mstcnv', t['mstcnv_stats'])
print('radiation replay', t.get('radiation_interface'), t.get('radiation_sentence'))
print('ocean_detail', {k: round(v, 2) for k, v in t['ocean_detail_seconds'].items()})
print('handoff bytes/arrays', t['handoff_bytes'], t['handoff_arrays'], 'melt_si device==numpy', t['melt_si_device_vs_numpy']['all_equal'])
print('cold counters', r[0]['counters_totals'])
print('cold libimf', r[0]['libimf_callbacks']['total'])
s = d.get('server_run')
if s:
    print(s['radiation_sentence']); print(s['radiation_interface']); print(s['server']); print(s['server_vs_replay_end_state'])
    print(s['server_outputs_vs_recorded_server_output']); print('left', s['model_processes_left_after_stop']); print('libimf server', s['libimf_callbacks']['total'])
    print('server stage table', [(a['name'], round(a['seconds'], 2)) for a in s['stage_table'] if a['calls']])
print(d['recorded_inputs_registry']['summary'])
for it in d['recorded_inputs_registry']['items']:
    if it['read_count']:
        print('  ', it['name'], it['role'], it['bytes_read'], it['source_file'][:90])
print('io audit files', len(d['io_audit']), sum(x['bytes'] or 0 for x in d['io_audit']))
print('stage table (timed run)', [(a['name'], a['kind'], round(a['seconds'], 2), None if a['share'] is None else round(100 * a['share'], 1)) for a in t['stage_table']])
