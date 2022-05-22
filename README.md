# Physics_Maths_and_others
Python scripts to do some physic simulations or math applications.


**Maths:**
- Function estimation by polynoms
- 2D grid transform
- some basics integration schemes (Euler, RK4)

**Physics:**
- weighing pendulum
- double weighing pendulum
- heat transfer in one dimension (x-axis) in a rod


**Next ideas:**
- triple (and n) weighing pendulum
- heat transfer in 2D (and in 3D)
- An autonomeous car that drive itself around a circuit
- circulation simulation around a traffic light
- plasma simulation


---
To check linting and typing, use the following command:
```
pylint --rcfile=.pylintrc ./maths ./physics
mypy --config-file=mypy.ini ./maths ./physics
```

---

To test folder, use the following command (wip):
```
pytest test
```
To see test coverage, use (need pytest-cov):
```
pytest test/ --cov-report term-missing
```



