#include "stdio.h"
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#define N 992 // Number of stars
#define G 6.674e-11 // Gravitational constant
#define m 2e30 // Mass of sun
#define ly 9e12
#define time_step 1000000 // Time step size
#define PI 3.14
#define simulation_time 120
double start_time, end_time, run_time;
int counter = 0 ,size, rank;


typedef struct {
    double pos_x;
    double pos_y;
    double vel_x;
    double vel_y;
} Star;

double star_rand(){
    double pos;
    double e_0= (rand() %1000);// / 1000);
    double e_1= (rand() %1000);// / 1000);
    double e_2= (rand() %1000);// / 1000);
    double e_3= (rand() %1000);// / 1000);
    pos = e_0 + e_1 * 1000 + e_2 * 1000000 + e_3 * 1000000000;
    return pos;
}

void init_stars(Star* proc_stars,int proc_size){
    for(int i=0;i<proc_size;i++){
        proc_stars[i].pos_x =  star_rand();
        proc_stars[i].pos_y =  star_rand();
        int vel = rand() % 200 + 100;
        double angle = (rand() % 1000) * 1000 * 2 * PI;
        proc_stars[i].vel_x =   vel * cos(angle);
        proc_stars[i].vel_y =   vel * sin(angle);
    }

}
void calc_vel(Star this_star, Star other_star){
    double x_dist = this_star.pos_x - other_star.pos_x, y_dist = this_star.pos_y - other_star.pos_y;
    double r_pow = pow((x_dist), 2) + pow((y_dist), 2);
    double angle = atan(y_dist/x_dist);
    if ((r_pow == 0) || (angle == 0)) return;
    double acce = G  * m / r_pow;

    this_star.vel_x = this_star.vel_x + acce*cos(angle) * time_step;
    this_star.vel_y = this_star.vel_y + acce*sin(angle) * time_step;
}

void update_stars(Star* proc_stars, Star* all_stars, int proc_size){
    for (int i=0; i<proc_size; i++){
        for (int j=0; j<N; j++){
            calc_vel(proc_stars[i], all_stars[j]);
        }
        proc_stars[i].pos_x = fmod((proc_stars[i].pos_x + proc_stars[i].vel_x * time_step), (100* ly));
        proc_stars[i].pos_y = fmod((proc_stars[i].pos_y + proc_stars[i].vel_y * time_step), (100* ly));
    }
}

void update_image(Star* all_stars, char image_num){
    FILE* file;
    char image_name[] = "image_#.csv";
    image_name[6] = image_num;
    file = fopen(image_name, "w+");
    for (int i = 0; i<N; i++)
        fprintf(file,"%lf,%lf\n",all_stars[i].pos_x, all_stars[i].pos_y);
    fclose(file);
}

int main(int argc, char** argv) {

    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    //MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name,&namelen);

    int proc_size = N/size; //every process own its stars
    Star* proc_stars = (Star*)malloc(proc_size*sizeof(Star));
    Star* all_stars = (Star*)malloc(N*sizeof(Star));


    int seeds[size];  //Array of seeds

    if (rank == 0) {
        srand(time(NULL)); // seed the random number generator only on Rank 0
        for (int i = 0; i < size; i++) {
            seeds[i] = rand(); // generate a random seed for each process
        }
    }

    MPI_Bcast(seeds, size, MPI_INT, 0, MPI_COMM_WORLD); // broadcast the array of random seeds to all processes
    srand(seeds[rank]); // seed the random number


    init_stars(proc_stars, proc_size);

    MPI_Allgather(proc_stars,proc_size*sizeof(Star),MPI_BYTE,all_stars,proc_size*sizeof(Star),MPI_BYTE,MPI_COMM_WORLD);

    if(rank == 0) //the "main" process take the current time and document the beginning of the "galaxy"
    {
        start_time = MPI_Wtime();
        update_image(all_stars,'0');

    }
    MPI_Bcast(&start_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);



    while (counter<1200){
        counter += 1;
        update_stars(proc_stars, all_stars,proc_size);
        MPI_Allgather(proc_stars,proc_size*sizeof(Star),MPI_BYTE, all_stars, proc_size*sizeof(Star),MPI_BYTE,MPI_COMM_WORLD);
        if(counter==600 && rank==0)
            {update_image(all_stars,'1');}
    }


    if(rank == 0)  //the "main" process calculate the running time and document the end of the "galaxy"
    {
        end_time =MPI_Wtime();
        run_time =end_time-start_time;
        printf("run time: %lf\n",run_time);
        update_image(all_stars,'2');
    }

    //free all the program demends & dynamic allocations
    free(all_stars);
    free(proc_stars);
    MPI_Finalize();
    return 0;















    return 0;

}