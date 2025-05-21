🔹 1. Introduction
In today’s computing landscape, heterogeneous systems—composed of processors with varying capabilities—are the backbone of high-performance computing. Efficient scheduling of independent tasks on such systems is critical to optimizing performance, minimizing execution time, and maximizing resource utilization.
This project focuses on implementing and analyzing a variety of heuristic-based scheduling algorithms tailored for independent task allocation in heterogeneous environments.

🔹 2. Objective
Develop and compare multiple task scheduling algorithms.

Minimize the makespan (total time to complete all tasks).

Achieve better load balancing among heterogeneous processors.

Handle task and processor heterogeneity using customized datasets.

Improve upon basic heuristics using priority-based enhancements.

🔹 3. Scheduling Algorithms Implemented
🔸 a. Min-Min Algorithm
Selects the task with the minimum expected completion time (ECT).

Assigns it to the processor that can execute it fastest.

Works best for tasks with uniform execution times.

🔸 b. Max-Min Algorithm
Chooses the task with the maximum ECT among the minimum ECTs for all tasks.

Prioritizes long tasks early, reducing their delay.

🔸 c. Sufferage Algorithm
Calculates the difference between the best and second-best processor times for each task.

The task with the highest “sufferage value” is scheduled first.

🔸 d. List Sufferage
Extends the Sufferage approach by organizing tasks in a list.

Improves decision-making when multiple tasks have the same sufferage.

🔸 e. Priority-Based Scheduling (PB)
Assigns priorities based on factors like task size and processor speed.

Dynamically schedules tasks using these priorities.

🔸 f. Tenacious Priority-Based (TPB)
Builds on PB but adds retry logic, backoff penalties, and resilience mechanisms.

Ensures even low-priority tasks eventually get scheduled without starvation.

🔹 4. Datasets Used
Two ETC (Expected Time to Compute) matrices were used:

512 x 16: 512 tasks on 16 processors.

1024 x 32: 1024 tasks on 32 processors.

🧬 Dataset Types (Heterogeneity Levels):
Task Heterogeneity:

lo (low variation)

hi (high variation)

Processor Heterogeneity:

lo (homogeneous)

hi (diverse)

Thus, combinations like hilo, lolo, hihi, lohi represent system complexity levels.

Also includes:

auc – Application Use Case–driven

buc – Benchmark Use Case–driven

🔹 5. Evaluation Metrics
✅ Makespan
Total time to execute all tasks. The lower, the better.

✅ Load Balance
Measures even distribution of workload.

Better load balance → reduced processor idle time.

✅ Processor Utilization
% of time processors were actually executing tasks.

Shows system efficiency.

✅ Throughput
Total number of tasks completed per time unit.

🔹 6. Complexity Analysis
| Algorithm      | Time Complexity (approx)   |
| -------------- | -------------------------- |
| Min-Min        | O(n²·m)                    |
| Max-Min        | O(n²·m)                    |
| Sufferage      | O(n²·m)                    |
| List Sufferage | O(n²·m)                    |
| PB             | O(n·logn) + heuristic cost |
| TPB            | O(n·logn) + penalty cycles |


🔹 7. Results Summary
The TPB algorithm consistently outperformed others in:

Minimizing makespan.

Maintaining balanced load.

Ensuring fairness in scheduling.

Min-Min and Max-Min were faster but lacked flexibility in heterogeneous conditions.

🔹 8. Conclusion
This project demonstrates that basic heuristics like Min-Min and Max-Min are fast but not always optimal for heterogeneous systems. Priority-based enhancements, especially TPB, offer more robust and fair task scheduling by adapting dynamically to system constraints.
Such strategies are crucial for real-world workloads where resource availability and task demands vary significantly.

🔹 9. Future Work
Add support for task dependencies (DAGs).

Explore metaheuristics like Genetic Algorithm or Ant Colony Optimization.

Visualize scheduling in real-time (Gantt charts).

Integrate energy-aware scheduling models.

🔹 10. References
Braun, T.D., Siegel, H.J., et al. “A Comparison of Eleven Static Heuristics for Mapping a Class of Independent Tasks onto Heterogeneous Distributed Computing Systems.” Journal of Parallel and Distributed Computing, 2001.

Kwok, Y.-K., & Ahmad, I. (1999). “Static Scheduling Algorithms for Allocating Directed Task Graphs to Multiprocessors.” ACM Computing Surveys.
