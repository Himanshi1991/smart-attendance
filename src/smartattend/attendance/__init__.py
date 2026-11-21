"""Face-recognition attendance: enrol a roster, recognise faces per frame, aggregate a register."""
from smartattend.attendance.roster import Roster
from smartattend.attendance.session import AttendanceSession, iter_frames

__all__ = ["Roster", "AttendanceSession", "iter_frames"]
