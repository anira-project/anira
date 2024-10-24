#include <signal.h>
#include "JackClient.h"

JackClient* jackClient;

static void signal_handler ( int sig ) {
    fprintf ( stderr, "signal received, exiting ...\n" );
    exit (0);
}

void cleanup () {
    delete jackClient;
    std::cout << "Cleanup done!" << std::endl;
}

int main ( int argc, char *argv[] ) {

    // Initialize jack client. No explicit call to jackClient.prepare() is needed, since the registered samplerate callback does that for us.
    jackClient = new JackClient(argc, argv);
    std::cout << "[info] initialized jack client" << std::endl;

    /* install a signal handler to properly quit the jack client */
#ifdef WIN32
    signal ( SIGINT, signal_handler );
    signal ( SIGABRT, signal_handler );
    signal ( SIGTERM, signal_handler );
#else
    signal ( SIGQUIT, signal_handler );
    signal ( SIGTERM, signal_handler );
    signal ( SIGHUP, signal_handler );
    signal ( SIGINT, signal_handler );
#endif

    atexit(cleanup);
    
    /* keep running until the transport stops */

    while (1)
    {
#ifdef WIN32
        Sleep ( 1000 );
#else
        sleep ( 1 );
#endif
    }

    exit ( 0 );
}