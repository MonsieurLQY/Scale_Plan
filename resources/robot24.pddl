(define (domain robot24 )
  (:requirements :strips :typing :negative-preconditions) 
  (:types robot object)
  (:predicates
    (at ?robot - robot ?object - object)
    (inaction ?robot - robot) 
    (at-location  ?object - object ?location - object)
    (switch-on ?robot - robot ?object - object)
    (switch-off ?robot - robot ?object - object)
  )
  
  (:action GoToObject
    :parameters (?robot - robot ?object - object)
    :precondition (not (inaction ?robot))
    :effect (and 
      (at ?robot ?object)
      (forall (?another_object - object)
        (when (at ?robot ?another_object)
          (not (at ?robot ?another_object))
        )
      )
      (not (inaction ?robot))
    )
  )


  (:action SwitchOn
    :parameters (?robot - robot ?object - object)
    :precondition (and 
      (not (inaction ?robot))
      (at ?robot ?object)
    )   
    :effect (and
      (not (inaction ?robot))
      (switch-on ?robot ?object)
    ) 
  )

  (:action SwitchOff
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (switch-off ?robot ?object)
    )    
  )

)