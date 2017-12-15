#ifndef _STOPWATCH_H
#define _STOPWATCH_H

// includes, system
#include <ctime>
#include <sys/time.h>

// Note: This is currently Linux-specific!
class Stopwatch
{
    //! Start of measurement
    struct timeval start_time;

    //! Time difference between the last start and stop
    double  diff_time;

    //! TOTAL time difference between starts and stops
    double  total_time;

    //! flag if the stop watch is running
    bool running;

    //! Number of times clock has been started
    //! and stopped to allow averaging
    int clock_sessions;

public:
    //! Constructor, default
    Stopwatch() : diff_time(0), total_time(0), running(false), clock_sessions(0) {}

    // Destructor
    //~Stopwatch();

    //! Start time measurement
    inline void start();

    //! Stop time measurement
    inline void stop();

    //! Reset time counters to zero
    inline void reset();

    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned, otherwise the
    //! time between the last start() and stop call is returned
    inline double getTime() const;

    //! Added by G.Vernardos.
    //! Set the time to selected value
    inline void setTime(double time0);

    //! Mean time to date based on the number of times the stopwatch has been 
    //! _stopped_ (ie finished sessions) and the current total time
    inline double getAverageTime() const;

private:
    // helper functions
  
    //! Get difference between start time and current time
    inline double getDiffTime() const;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void
Stopwatch::start() {

  gettimeofday( &start_time, 0);
  running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void
Stopwatch::stop() {

  diff_time = getDiffTime();
  total_time += diff_time;
  running = false;
  clock_sessions++;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does 
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void
Stopwatch::reset() 
{
  diff_time = 0;
  total_time = 0;
  clock_sessions = 0;
  if( running )
    gettimeofday( &start_time, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the 
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline double 
Stopwatch::getTime() const 
{
    // Return the TOTAL time to date
    double retval = total_time;
    if( running) {

        retval += getDiffTime();
    }

    return retval/(double)1000.0;
}
////////////////////////////////////////////////////////////////////////////////
//! Added by G.Vernardos
//! Function to add given time(in sec) to the total time(in msec)
////////////////////////////////////////////////////////////////////////////////
inline void
Stopwatch::setTime(double time0)
{
  total_time += time0*1000.0;
}
////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline double 
Stopwatch::getAverageTime() const
{
    return total_time/clock_sessions;
}



////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
inline double
Stopwatch::getDiffTime() const 
{
  struct timeval t_time;
  gettimeofday( &t_time, 0);

  // time difference in milli-seconds
  return  (double) (1000.0 * ( t_time.tv_sec - start_time.tv_sec) 
                + (0.001 * (t_time.tv_usec - start_time.tv_usec)) );
}

#endif // _STOPWATCH_H
