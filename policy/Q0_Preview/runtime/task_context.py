"""Human-readable task clarification; no runtime reward/pose access."""
import json
from pathlib import Path
import re

TASK_INFORMATION_MODES = ('original', 'official_only', 'strict_official')
ASSETS = Path(__file__).parent.parent / 'assets/tasks'


def information_mode(value='original'):
    if value not in TASK_INFORMATION_MODES:
        raise ValueError(f'Unknown task_information_mode: {value!r}')
    return value


def task_card(name, mode='original'):
    mode = information_mode(mode)
    if mode in ('official_only', 'strict_official'):
        data = json.loads((ASSETS/'official_task_descriptions.json').read_text())
        if name not in data['cards']:
            raise ValueError(f'No frozen official description for {name!r}; refusing fallback')
        card = data['cards'][name]
        return dict(requirements=(card['description'] if mode == 'strict_official' else
                                  card['description'] + '\n\nPublic scoring:\n' + '\n'.join(card['scoring'])),
                    finish='', return_home_required=card['return_home_required'],
                    control_step_limit=card['control_step_limit'])
    return json.loads((ASSETS/'task_cards.json').read_text())['cards'][name]


def task_step_limit(name, configured_limit=None, mode='original'):
    """Static task budget audited offline against the frozen benchmark sources."""
    try:
        limit = task_card(name, mode)['control_step_limit']
    except (KeyError, TypeError) as exc:
        raise ValueError(f'No audited control-step limit for task {name!r}') from exc
    if type(limit) is not int or limit <= 0:
        raise ValueError(f'Invalid audited control-step limit for task {name!r}')
    if configured_limit is not None and (type(configured_limit) is not int or configured_limit != limit):
        raise ValueError(f'Configured max_control_steps={configured_limit!r} disagrees with '
                         f'task {name!r} limit {limit}; update the audited task metadata explicitly')
    return limit


def task_reference(name, mode='original'):
    """Return only this task's public definitions, never live reward or object state."""
    if information_mode(mode) in ('official_only', 'strict_official'):
        raise ValueError('Task reference is disabled in official_only mode')
    data = json.loads((ASSETS/'task_reference.json').read_text())
    if name not in data['cards']:
        raise ValueError(f'No audited task reference for {name}')
    card = data['cards'][name]
    return {'task': name, 'notice': data['notice'],
            'completion_details': card['completion_details'], 'process_rules': card['process_rules'],
            'assessment_notes': card.get('assessment_notes', '')}


def task_context(name, benchmark_root=None, verify=False, mode='original'):
    """Static task cards; never read live task/reward source or state."""
    card = task_card(name, mode)
    if information_mode(mode) in ('official_only', 'strict_official'):
        return card['requirements']
    reference = task_reference(name)
    text = card['requirements'] + '\n' + card['finish']
    for title, key in (('Completion details', 'completion_details'),
                       ('Process rules', 'process_rules'), ('Evaluation notes', 'assessment_notes')):
        extra = []
        for sentence in re.split(r'(?<=[.!?])\s+', reference[key].strip()):
            # Omit inherited placeholders and facts already stated in the brief.
            if (sentence and not sentence.startswith('No additional ')
                    and sentence not in text and sentence not in extra):
                extra.append(sentence)
        if extra:
            text += '\n' + title + ': ' + ' '.join(extra)
    return text


def stage_task_reference(work, name, mode='original'):
    """Only original runs expose the restricted reference; reject stale workspaces."""
    path = Path(work)/'task_reference.json'
    if information_mode(mode) in ('official_only', 'strict_official'):
        if path.exists() or path.is_symlink():
            raise RuntimeError('Restricted reference in official_only workspace; use a fresh workspace')
        return
    path.write_text(json.dumps(task_reference(name), indent=2))


def planning_prompt(text, mode='original'):
    """Remove the reference-file and task-strategy instructions from the strict arm."""
    if information_mode(mode) == 'original':
        return text
    if information_mode(mode) == 'strict_official':
        return (ASSETS.parent/'prompts/strict_prompt.md').read_text()
    start = text.index('Read `instruction` and `task_requirements`.')
    end = text.index('\n## Gripper pose reference points', start)
    return (text[:start] + 'Read `instruction` and `task_requirements`. The latter contains only the '
            'official task description and its public scoring milestones. Use current observations '
            'and the general action tools to plan and recover. No additional task reference is supplied.\n'
            + text[end:])
