"""Interaction must remain usable during transitions and rapid reversals."""
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QLineEdit, QPushButton
from vstudio_ui.motion import PageMotion, InteractionMotion, reveal


def test_navigation_keeps_forms_and_accepts_input_during_transition(qapp, monkeypatch):
    monkeypatch.delenv('VOCALSTUDIO_REDUCED_MOTION', raising=False)
    stack = QStackedWidget()
    stack.resize(500, 300)
    pages = []
    for index in range(3):
        page = QWidget()
        layout = QVBoxLayout(page)
        edit = QLineEdit(str(index)); layout.addWidget(edit)
        stack.addWidget(page); pages.append(edit)
    motion = PageMotion(stack)
    stack.show(); QTest.qWait(30)
    motion.switch_to(1)
    assert stack.currentIndex() == 1 and motion.frames
    pages[1].setFocus(); QTest.keyClicks(pages[1], ' retained')
    motion.switch_to(2); motion.switch_to(1)
    QTest.qWait(400)
    assert pages[1].text() == '1 retained'
    assert not motion.isVisible() and motion.frames is None
    motion.switch_to(0); stack.resize(520, 320); qapp.processEvents()
    assert motion.frames is None
    stack.close()


def test_reveal_fast_reversal_restores_fixed_height_and_text(qapp, monkeypatch):
    monkeypatch.delenv('VOCALSTUDIO_REDUCED_MOTION', raising=False)
    root = QWidget(); layout = QVBoxLayout(root)
    edit = QLineEdit('keep my input'); edit.setFixedHeight(80); layout.addWidget(edit)
    root.show(); QTest.qWait(20)
    reveal(edit, False); QTest.qWait(40); reveal(edit, True)
    QTest.qWait(300)
    assert edit.isVisible() and edit.minimumHeight() == edit.maximumHeight() == 80
    assert edit.text() == 'keep my input'
    reveal(edit, False); QTest.qWait(300)
    assert edit.isHidden() and edit.minimumHeight() == 80
    root.close()


def test_reduced_motion_and_feedback_do_not_swallow_clicks(qapp, monkeypatch):
    root = QWidget(); layout = QVBoxLayout(root)
    button = QPushButton('Run'); layout.addWidget(button)
    controller = InteractionMotion(root)
    clicked = []; button.clicked.connect(lambda: clicked.append(True))
    root.show(); QTest.qWait(20)
    QTest.mouseClick(button, Qt.LeftButton)
    assert clicked == [True] and hasattr(button, '_motion_feedback')
    monkeypatch.setenv('VOCALSTUDIO_REDUCED_MOTION', '1')
    other = QPushButton('Cancel'); layout.addWidget(other); qapp.processEvents()
    QTest.mouseClick(other, Qt.LeftButton)
    assert not hasattr(other, '_motion_feedback')
    root.close()
