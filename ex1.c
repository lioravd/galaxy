#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define N 992 // Number of stars
#define G 6.674e-11 // Gravitational constant
#define MASS 2e30 // Mass of sun
#define DOMAIN_SIZE 100.0 // Domain size in light years
#define TIME_STEP 0.01 // Time step size

typedef struct {
    double x;
    double y;
} Vector2D;

typedef struct {
    Vector2D position;
    Vector2D velocity;
} Star;

void initialize_stars(Star* stars, int num_stars, double domain_size, double avg_speed) {
    srand(time(NULL));

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = num_stars / size;

    for (int i = 0; i < chunk_size; i++) {
        int index = rank * chunk_size + i;

        // Generate random positions
        stars[index].position.x = ((double)rand() / RAND_MAX) * domain_size;
        stars[index].position.y = ((double)rand() / RAND_MAX) * domain_size;

        // Generate random velocities
        double speed = avg_speed * (0.5 + ((double)rand() / RAND_MAX));
        double angle = 2 * M_PI * ((double)rand() / RAND_MAX);
        stars[index].velocity.x = speed * cos(angle);
        stars[index].velocity.y = speed * sin(angle);
    }
}

void update_stars(Star* stars, int num_stars, double time_step) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = num_stars / size;

    for (int i = 0; i < chunk_size; i++) {
        int index = rank * chunk_size + i;

        Vector2D acceleration = {0.0, 0.0};

        for (int j = 0; j < num_stars; j++) {
            if (j != index) {
                Vector2D r = {
                        stars[j].position.x - stars[index].position.x,
                        stars[j].position.y - stars[index].position.y
                };

                double distance_squared = r.x * r.x + r.y * r.y;
                double distance_cubed = distance_squared * sqrt(distance_squared);

                Vector2D force = {
                        G * MASS * r.x / distance_cubed,
                        G * MASS * r.y / distance_cubed
                };

                acceleration.x += force.x;
                acceleration.y += force.y;
            }
        }

        // Update position and velocity
        stars[index].position.x += stars[index].velocity.x * time_step;
        stars[index].position.y += stars[index].velocity.y * time_step;

        stars[index].velocity.x += acceleration.x * time_step;
        stars[index].velocity.y += acceleration.y * time_step;
    }
}

void simulate_n_body(int num_steps) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;
    int num_stars = chunk_size * size;

    Star* stars = (Star*)malloc(num_stars * sizeof(Star));

    initialize_stars(stars, num_stars, DOMAIN_SIZE, 200000.0);

    for (int step = 0; step < num_steps; step++) {
        update_stars(stars, num_stars, TIME_STEP);

        // Gather all the stars' data to process 0
        Star* all_stars = NULL;
        if (rank == 0) {
            all_stars = (Star*)malloc(N * sizeof(Star));
        }
        MPI_Gather(stars, chunk_size * sizeof(Star), MPI_BYTE, all_stars, chunk_size * sizeof(Star), MPI_BYTE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Process and visualize the data (not shown in this code snippet)
            // ...
            // ...
            free(all_stars);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(stars);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start_time = MPI_Wtime();

    simulate_n_body(6000); // Run simulation for 2 minutes (6000 steps with 0.02 sec per step)

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();

    return 0;
}
