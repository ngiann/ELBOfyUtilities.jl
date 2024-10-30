function get_callback_and_track_elbo_for_tracking_test_evidence(elbo, iteration_test)

    trackelbo = trackElbo(elbo)

    function cb(_)

        incrementcounter!(trackelbo)
        
        if iteration_test > 0 && mod(getcounter(trackelbo), iteration_test) == 1

            testlogevidence = testelbo(trackelbo, trackelbo.bestsolutionsofar)
            
            @printf("\t Test evidence is %f\n", testlogevidence)

        end
        
        false
        
    end

    return trackelbo, cb

end