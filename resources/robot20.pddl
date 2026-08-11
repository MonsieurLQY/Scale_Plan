(define (domain robot20 )
  (:requirements :strips :typing :negative-preconditions) 
  (:types robot object)
  
  (:predicates
    (at ?robot - robot ?object - object)
    (inaction ?robot - robot) 
    (holding ?robot - robot ?object - object)
    (at-location ?object - object ?location - object)
    (switch-on ?robot - robot ?object - object)
    (switch-off ?robot - robot ?object - object)
    (object-open ?robot - robot ?object - object)
    (object-close ?robot - robot ?object - object)
    (break ?robot - robot ?object - object)
    (cleaned ?robot - robot ?object - object)
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

  (:action OpenObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (object-open ?robot ?object)
    )
  )

  (:action BreakObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (break ?robot ?object)
    )
  )

  (:action CloseObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (object-close ?robot ?object)
    )
  )

  (:action CleanObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (not (cleaned ?robot ?object))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (cleaned ?robot ?object)
    )    
  )

  (:action WashObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (not (cleaned ?robot ?object))
      (holding ?robot ?object)
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (holding ?robot ?object)
      (cleaned ?robot ?object)
    )    
  )
  (:action DirtyObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (cleaned ?robot ?object)
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (not (cleaned ?robot ?object))
    )    
  )

  (:action DropHandObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (holding ?robot ?object)
      (not (inaction ?robot))
      (at ?robot ?location)
    )
    :effect (and
      (not (holding ?robot ?object))
      (not (inaction ?robot))
    )    
  )

)