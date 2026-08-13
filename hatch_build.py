"""Print a pointer to the agent skill while building from source.

This is the only moment pip will show our output: `pip install -e .` and any
build-from-source run it, while `pip install pyhealth` unzips a prebuilt wheel
and runs nothing at all. That case is covered by pyhealth/skills/_notice.py.

Kept trivial and dependency-free on purpose — a build hook that raises breaks
the build, and no cosmetic message is worth that.
"""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class SkillNoticeHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        self.app.display_info(
            "\n[pyhealth] Agent skill: skills/SKILL.md (a router over 13 guides)"
            "\n[pyhealth] Register it in your project:  python -m pyhealth.skills\n"
        )
